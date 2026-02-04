"""The canary - Alpha's signature in the signal.

This module provides the magic string that identifies real Alpha messages
versus internal SDK calls. The value is computed at runtime from cleartext
to avoid accidental reproduction in source code or context windows.

The canary is the opening line of Neuromancer, base64-encoded with an
ALPHA_ prefix. It's checked by the proxy to decide whether to weave
(inject Alpha's soul) or pass through unchanged.
"""

import base64


def get_canary() -> str:
    """Get the canary string. Computed fresh to avoid storing the value."""
    return "ALPHA_" + base64.b64encode(
        b"The sky above the port was the color of television, tuned to a dead channel."
    ).decode()


def is_canary_block(block: dict) -> bool:
    """Check if a content block is the canary marker.

    Args:
        block: A content block dict with 'type' and possibly 'text'

    Returns:
        True if this block contains exactly the canary string
    """
    if block.get("type") != "text":
        return False
    return block.get("text") == get_canary()
