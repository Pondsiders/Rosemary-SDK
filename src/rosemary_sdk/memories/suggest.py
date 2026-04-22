"""Memory suggestion - what's memorable from this turn?

After each turn completes, asks the chat model what moments are worth
remembering. Results are returned to the caller (RosemaryClient) for
injection on the next turn.

This is fire-and-forget - call it as an asyncio task after turn completes.
"""

import json
import os

import logfire

from ..inference_client import get_client
from ..prompts import load_prompt

# Configuration from environment — crash at import time if not set
CHAT_MODEL = os.environ["CHAT_MODEL"]


def _parse_memorables(text: str) -> list[str]:
    """Parse JSON array of strings from model output."""
    if not text:
        return []

    text = text.strip()

    # Find JSON array in output
    start = text.find("[")
    end = text.rfind("]") + 1

    if start == -1 or end == 0:
        logfire.warning("No JSON array found in suggest output", raw=text[:200])
        return []

    try:
        result = json.loads(text[start:end])
        if isinstance(result, list):
            return [s.strip() for s in result if isinstance(s, str) and s.strip()]
        return []
    except json.JSONDecodeError as e:
        logfire.warning("Failed to parse suggest JSON", error=str(e), raw=text[:200])
        return []


async def _call_model(user_content: str, assistant_content: str) -> list[str]:
    """Ask the chat model what's memorable from this turn."""
    system_prompt = load_prompt("suggest-system")
    turn_template = load_prompt("suggest-turn")

    user_prompt = turn_template.format(
        user_content=user_content[:2000],
        assistant_content=assistant_content[:4000],
    )

    with logfire.span(
        "suggest.model",
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "openai",
            "gen_ai.request.model": CHAT_MODEL,
        }
    ) as span:
        try:
            # Gemma 3 non-thinking sampling (per Unsloth model card).
            response = await get_client().chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                top_p=0.95,
                extra_body={
                    "top_k": 64,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                },
                timeout=30.0,
            )

            output = response.choices[0].message.content or ""

            if response.usage:
                span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
            span.set_attribute("gen_ai.response.model", CHAT_MODEL)

            memorables = _parse_memorables(output)
            logfire.info("Suggest memorables extracted", count=len(memorables))
            return memorables

        except Exception as e:
            logfire.error("Suggest failed", error=str(e))
            return []


async def suggest(user_content: str, assistant_content: str, session_id: str) -> list[str]:
    """
    Extract memorables from a turn.

    Fire-and-forget: call as an asyncio task after turn completes.
    Returns the list of memorable strings for the caller to hold.

    Args:
        user_content: What the user said this turn
        assistant_content: What the AI said this turn
        session_id: Current session ID

    Returns:
        List of memorable strings (empty if nothing notable)
    """
    with logfire.span("suggest", session_id=session_id[:8] if session_id else "none"):
        memorables = await _call_model(user_content, assistant_content)

        if memorables:
            logfire.info("Memorables extracted", session_id=session_id[:8], count=len(memorables))
        else:
            logfire.debug("No memorables this turn")

        return memorables
