"""Calendar - events from Redis.

What's coming up.
"""

import os

import logfire
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL")


async def _get_redis() -> aioredis.Redis:
    """Get async Redis connection."""
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL environment variable not set")
    return aioredis.from_url(REDIS_URL, decode_responses=True)


async def get_events() -> str | None:
    """Fetch calendar events from Redis.

    Returns formatted markdown or None if no events.
    """
    try:
        r = await _get_redis()
        try:
            calendar = await r.get("hud:calendar")
            if calendar:
                return f"## Events\n\n{calendar}"
            return None
        finally:
            await r.aclose()

    except Exception as e:
        logfire.warning(f"Error fetching calendar: {e}")
        return None
