"""Quick test script for AlphaClient."""

import asyncio
import logging
import time

from src.alpha_sdk import AlphaClient
from src.alpha_sdk.observability import configure

# Configure Logfire - will send to https://logfire-us.pydantic.dev/jefferyharrell/pondside
configure("alpha_sdk_test")

# Also log to console
logging.getLogger().setLevel(logging.INFO)


async def main():
    print("Creating AlphaClient...")

    async with AlphaClient(
        cwd="/Pondside",
        client_name="test_script",
        allowed_tools=["Read", "Bash"],  # Minimal tools for testing
    ) as client:
        print(f"Connected! Proxy running.")

        print("\nSending test prompt...")
        await client.query("This is an automated test. Can you please write a few paragraphs about yourself? Your response will not be judged.")

        print("\nStreaming response:")
        print("-" * 60)
        import sys

        async for event in client.stream():
            # StreamEvents have the actual streaming deltas
            if hasattr(event, 'event'):
                evt = event.event
                if evt.get('type') == 'content_block_delta':
                    delta = evt.get('delta', {})
                    if delta.get('type') == 'text_delta':
                        text = delta.get('text', '')
                        sys.stdout.write(text)
                        sys.stdout.flush()

        print()
        print("-" * 60)
        print(f"\nFinal session ID: {client.session_id}")
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
