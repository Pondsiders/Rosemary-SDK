"""Prompt loading from the plugin directory.

All personality-bearing prompts live in Rosemary-Plugin/prompts/.
The SDK loads them at startup. If a prompt file is missing, we
either use a sensible default or raise a clear error.

This is the boundary between machinery and soul. The SDK doesn't
know or care what the prompts say — it just loads them and injects
them where they belong. The personality lives in the plugin.

Prompts are cached after first load. To reload, call clear_cache().
"""

import os
from pathlib import Path

import logfire

# Plugin directory — sibling of the SDK package, or overridden by env var
_PLUGIN_DIR = Path(os.environ.get(
    "ROSEMARY_PLUGIN_DIR",
    str(Path(__file__).parent.parent.parent.parent / "Rosemary-Plugin"),
))
PROMPTS_DIR = _PLUGIN_DIR / "prompts"

# Cache loaded prompts
_cache: dict[str, str] = {}


def load_prompt(name: str, required: bool = True) -> str | None:
    """Load a prompt file from the plugin's prompts directory.

    Looks for PROMPTS_DIR/{name}.md. Caches after first read.

    Args:
        name: Prompt name (without .md extension)
        required: If True, raises FileNotFoundError when missing

    Returns:
        The prompt text (stripped), or None if optional and missing
    """
    if name in _cache:
        return _cache[name]

    path = PROMPTS_DIR / f"{name}.md"

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required prompt not found: {path}\n"
                f"Create this file in your Rosemary-Plugin/prompts/ directory."
            )
        logfire.debug(f"Optional prompt not found: {path}")
        return None

    text = path.read_text().strip()
    _cache[name] = text
    logfire.debug(f"Loaded prompt '{name}' ({len(text)} chars)")
    return text


def clear_cache() -> None:
    """Clear the prompt cache. Forces reload on next access."""
    _cache.clear()


def get_prompts_dir() -> Path:
    """Get the prompts directory path (for diagnostics)."""
    return PROMPTS_DIR
