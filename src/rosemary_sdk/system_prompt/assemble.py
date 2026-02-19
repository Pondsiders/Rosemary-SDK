"""System prompt assembly - weaving all threads together.

This is the main entry point for building Rosemary's complete system prompt.
Each piece is fetched and assembled into a coherent whole.

Note: Rosemary v0.1.0 has no routines, no calendar, no todos.
She gets: soul, capsules (from her own Postgres), here (client + weather),
and context files from her plugin directory. That's it.

Redis-dependent blocks (letter from last night, today so far, calendar,
todos) are intentionally excluded — those are Alpha's infrastructure.
If Rosemary gets her own routines later, add them back with namespaced keys.
"""

import asyncio

import logfire

from .soul import get_soul
from .capsules import get_capsules
from .here import get_here
from .context import load_context


async def assemble(client: str | None = None, hostname: str | None = None) -> list[dict]:
    """Assemble the complete Rosemary system prompt.

    Args:
        client: Client name (e.g., "rosemary-app")
        hostname: Machine hostname (auto-detected if not provided)

    Returns:
        List of system prompt blocks ready for the API.
        Each block is {"type": "text", "text": "..."}.
    """
    with logfire.span("assemble_system_prompt", client=client or "unknown") as span:
        # Fetch dynamic data in parallel
        capsules_task = get_capsules()
        here_task = get_here(client, hostname)

        (older_capsule, newer_capsule), here_block = await asyncio.gather(
            capsules_task,
            here_task,
        )

        span.set_attribute("has_older_capsule", bool(older_capsule))
        span.set_attribute("has_newer_capsule", bool(newer_capsule))

        # Load context files (sync operation, fast)
        context_blocks, context_hints = load_context()

        # Build the system blocks
        blocks = []

        # Soul - who I am
        blocks.append({"type": "text", "text": f"# Rosemary\n\n{get_soul()}"})

        # Capsules - what happened recently (from own Postgres)
        if older_capsule:
            blocks.append({"type": "text", "text": older_capsule})
        if newer_capsule:
            blocks.append({"type": "text", "text": newer_capsule})

        # Here - client, machine, weather
        blocks.append({"type": "text", "text": here_block})

        # Context files
        for ctx in context_blocks:
            blocks.append({
                "type": "text",
                "text": f"## Context: {ctx['path']}\n\n{ctx['content']}"
            })

        # Context hints
        if context_hints:
            hints_text = "## Context available\n\n"
            hints_text += "**BLOCKING REQUIREMENT:** When working on topics listed below, you MUST read the corresponding file BEFORE proceeding. Use the Read tool.\n\n"
            hints_text += "\n".join(f"- {hint}" for hint in context_hints)
            blocks.append({"type": "text", "text": hints_text})

        span.set_attribute("total_blocks", len(blocks))
        span.set_attribute("context_files", len(context_blocks))
        span.set_attribute("context_hints", len(context_hints))

        logfire.debug(f"Assembled system prompt: {len(blocks)} blocks")
        return blocks
