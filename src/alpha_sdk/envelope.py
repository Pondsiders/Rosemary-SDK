"""Envelope - wrapping user messages for transport.

Builds the structured input that carries user content, memories,
and metadata through the proxy. The canary at the end tells the
proxy "this is a real Alpha message, weave it."
"""

import json
from typing import Any

import pendulum
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .canary import get_canary


def format_memory_block(memory: dict) -> str:
    """Format a memory for inclusion as a content block.

    Creates human-readable memory text with relative timestamps.
    """
    mem_id = memory.get("id", "?")
    created_at = memory.get("created_at", "")
    content = memory.get("content", "").strip()
    score = memory.get("score")

    # Simple relative time formatting
    relative_time = created_at  # fallback
    try:
        dt = pendulum.parse(created_at)
        now = pendulum.now(dt.timezone or "America/Los_Angeles")
        diff = now.diff(dt)
        if diff.in_days() == 0:
            relative_time = f"today at {dt.format('h:mm A')}"
        elif diff.in_days() == 1:
            relative_time = f"yesterday at {dt.format('h:mm A')}"
        elif diff.in_days() < 7:
            relative_time = f"{diff.in_days()} days ago"
        elif diff.in_days() < 30:
            weeks = diff.in_days() // 7
            relative_time = f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            relative_time = dt.format("ddd MMM D YYYY")
    except Exception:
        pass

    # Include score if present (helps with debugging/transparency)
    score_str = f", score {score:.2f}" if score else ""
    return f"Memory #{mem_id} ({relative_time}{score_str}):\n{content}"


def build_envelope(
    prompt: str | list[Any],
    session_id: str | None,
    client_name: str = "alpha_sdk",
    memories: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Build structured input as a content array.

    Architecture: Content array with three sections:
    1. User content (text, images, whatever the user sent)
    2. Memory blocks (human-readable, permanent part of transcript)
    3. Canary block (tells proxy to weave - proxy removes this)

    Args:
        prompt: The user's message (string or list of content blocks)
        session_id: Current session ID
        client_name: Name of the client (for metadata)
        memories: Optional list of memories from recall

    Returns:
        List of content blocks ready for transport.write()
    """
    content_blocks: list[dict[str, Any]] = []

    # 1. Add user's actual content (text, images, whatever)
    if isinstance(prompt, str):
        content_blocks.append({"type": "text", "text": prompt})
    else:
        # Already a list of content blocks - copy to avoid mutating original
        content_blocks.extend(list(prompt))

    # 2. Add memory blocks (formatted, human-readable, permanent)
    if memories:
        for mem in memories:
            memory_text = format_memory_block(mem)
            content_blocks.append({"type": "text", "text": memory_text})

    # 3. Add the canary as the final block (proxy will strip this)
    content_blocks.append({"type": "text", "text": get_canary()})

    return content_blocks


def build_user_message(
    content_blocks: list[dict[str, Any]],
    session_id: str | None,
) -> dict[str, Any]:
    """Build a complete user message for transport.write().

    Args:
        content_blocks: The content array from build_envelope()
        session_id: Session ID for the message

    Returns:
        Dict ready to be JSON-serialized and written to transport
    """
    return {
        "type": "user",
        "message": {"role": "user", "content": content_blocks},
        "parent_tool_use_id": None,
        "session_id": session_id or "new",
    }
