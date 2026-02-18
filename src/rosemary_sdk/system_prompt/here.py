"""Here - where I am right now.

Narrative orientation, weather, astronomy.
Answers the question: what's the situation?
"""

import os
import socket

import logfire
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL")


def _get_narrative(client: str | None, hostname: str) -> str:
    """Get narrative orientation by composing client + machine."""
    parts = []
    if client:
        key = client if ":" not in client else client.split(":")[0]
        parts.append(f"You're in {key.title()}.")
    machine = f"Running on {hostname}."
    parts.append(machine)
    return " ".join(parts) if parts else f"You're on {hostname}."


async def _get_redis() -> aioredis.Redis:
    """Get async Redis connection."""
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL environment variable not set")
    return aioredis.from_url(REDIS_URL, decode_responses=True)


def get_hostname() -> str:
    """Get the current machine's hostname."""
    return socket.gethostname()


async def get_weather() -> str | None:
    """Fetch weather from Redis."""
    try:
        r = await _get_redis()
        try:
            weather = await r.get("hud:weather")
            return weather
        finally:
            await r.aclose()
    except Exception as e:
        logfire.warning(f"Error fetching weather: {e}")
        return None


async def get_here(client: str | None = None, hostname: str | None = None) -> str:
    """Build the Here section.

    Args:
        client: Client name (e.g., "duckpond", "solitude:full")
        hostname: Override hostname (defaults to socket.gethostname())

    Returns:
        Formatted markdown string for the ## Here section.
    """
    hostname = hostname or get_hostname()
    weather = await get_weather()

    parts = []
    parts.append(_get_narrative(client, hostname))
    if weather:
        parts.append(f"\n{weather}")

    return "## Here\n\n" + "\n".join(parts)
