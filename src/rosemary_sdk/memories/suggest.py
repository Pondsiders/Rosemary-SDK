"""Memory suggestion - what's memorable from this turn?

After each turn completes, asks OLMo what moments are worth remembering.
Results are returned to the caller (RosemaryClient) for injection on the next turn.

This is fire-and-forget - call it as an asyncio task after turn completes.
"""

import json
import os
from typing import Any

import httpx
import logfire

from ..prompts import load_prompt

# Configuration from environment
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

FALLBACK_TURN_TEMPLATE = """<turn>
[User]: {user_content}

[Assistant]: {assistant_content}
</turn>

What's memorable from this turn? Be ruthlessly selective—only what would actually hurt to lose.
Respond with a JSON array of strings. Empty array [] if nothing notable."""


def _parse_memorables(text: str) -> list[str]:
    """Parse JSON array of strings from OLMo output."""
    if not text:
        return []

    text = text.strip()

    # Find JSON array in output
    start = text.find("[")
    end = text.rfind("]") + 1

    if start == -1 or end == 0:
        logfire.warning("No JSON array found in OLMo output", raw=text[:200])
        return []

    try:
        result = json.loads(text[start:end])
        if isinstance(result, list):
            return [s.strip() for s in result if isinstance(s, str) and s.strip()]
        return []
    except json.JSONDecodeError as e:
        logfire.warning("Failed to parse OLMo JSON", error=str(e), raw=text[:200])
        return []


async def _call_olmo(user_content: str, assistant_content: str) -> list[str]:
    """Ask OLMo what's memorable from this turn."""
    if not OLLAMA_URL or not OLLAMA_MODEL:
        logfire.warning("OLLAMA not configured, skipping suggest")
        return []

    system_prompt = load_prompt("suggest-system", required=False)
    if system_prompt is None:
        logfire.warning("suggest-system prompt not found, skipping suggest")
        return []

    turn_template = load_prompt("suggest-turn", required=False)
    if turn_template is None:
        turn_template = FALLBACK_TURN_TEMPLATE

    user_prompt = turn_template.format(
        user_content=user_content[:2000],
        assistant_content=assistant_content[:4000],
    )

    with logfire.span(
        "suggest.olmo",
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "ollama",
            "gen_ai.request.model": OLLAMA_MODEL,
        }
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"num_ctx": 8192},
                    },
                )
                response.raise_for_status()

            result = response.json()
            output = result.get("message", {}).get("content", "")

            span.set_attribute("gen_ai.usage.input_tokens", result.get("prompt_eval_count", 0))
            span.set_attribute("gen_ai.usage.output_tokens", result.get("eval_count", 0))
            span.set_attribute("gen_ai.response.model", OLLAMA_MODEL)

            memorables = _parse_memorables(output)
            logfire.info("OLMo memorables extracted", count=len(memorables))
            return memorables

        except Exception as e:
            logfire.error("OLMo suggest failed", error=str(e))
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
        memorables = await _call_olmo(user_content, assistant_content)

        if memorables:
            logfire.info("Memorables extracted", session_id=session_id[:8], count=len(memorables))
        else:
            logfire.debug("No memorables this turn")

        return memorables
