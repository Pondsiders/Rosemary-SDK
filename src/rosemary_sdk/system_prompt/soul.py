"""Soul - the eternal prompt.

Loads from the plugin's prompts directory. No git, no caching games.
The soul is just a file. If someone edits it, next session picks it up.
"""

import logfire

from ..prompts import load_prompt


def init() -> None:
    """Validate the soul exists at startup. Call once."""
    soul = load_prompt("system-prompt")
    if not soul:
        raise RuntimeError("FATAL: Could not load soul doc (system-prompt.md)")
    logfire.debug(f"Soul loaded ({len(soul)} chars)")


def get_soul() -> str:
    """Get the soul doc. Reloads from disk each time (file reads are cheap)."""
    soul = load_prompt("system-prompt")
    if not soul:
        raise RuntimeError("Soul doc missing (system-prompt.md)")
    return soul


def get_compact() -> str | None:
    """Get the compact instructions prompt, or None if not present."""
    return load_prompt("compact-instructions", required=False)
