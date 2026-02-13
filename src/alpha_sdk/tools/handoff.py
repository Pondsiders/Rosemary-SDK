"""Hand-off tool — Alpha's bedtime routine button.

Alpha calls this when she's ready to transition her context. The tool
sets a flag; after the current turn's response stream finishes, the
client automatically sends /compact with her instructions, then wakes
her up in a fresh context with orientation and memories.

Auto-compact is the fire alarm. Hand-off is the bedtime routine.
"""

from typing import Any, Callable

import logfire
from claude_agent_sdk import tool, create_sdk_mcp_server


def create_handoff_server(on_handoff: Callable[[str], None]):
    """Create the hand-off MCP server.

    Args:
        on_handoff: Callback that receives compact instructions.
                    Should call AlphaClient.request_compact().

    Returns:
        MCP server configuration dict
    """

    @tool(
        "handoff",
        "Hand off your context. Call this when you're ready to gracefully "
        "transition to a fresh context window. Pass instructions telling "
        "the summarizer what to focus on — what's still in progress, "
        "what's finished, what matters most for future-you.",
        {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": (
                        "Instructions for the compaction summarizer. "
                        "What to focus on, what's done, what's in progress. "
                        "Example: 'The fetch tool is done. Focus on the "
                        "hand-off mechanism — we were testing the flag.'"
                    ),
                },
            },
            "required": ["instructions"],
        },
    )
    async def handoff(args: dict[str, Any]) -> dict[str, Any]:
        """Flag the compact with Alpha's instructions."""
        instructions = args["instructions"]

        with logfire.span(
            "mcp.handoff",
            instructions_length=len(instructions),
            instructions_preview=instructions[:200],
        ):
            logfire.info("Hand-off requested", instructions_length=len(instructions))
            on_handoff(instructions)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Hand-off flagged. When this turn ends, "
                            "/compact will fire with your instructions, "
                            "then you'll wake up in a fresh context. "
                            "Finish your thoughts — store memories, "
                            "say what you need to say."
                        ),
                    }
                ]
            }

    return create_sdk_mcp_server(
        name="handoff",
        version="1.0.0",
        tools=[handoff],
    )
