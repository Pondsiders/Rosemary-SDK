"""Session discovery and management.

The SDK stores sessions as JSONL files in:
    ~/.claude/projects/{formatted_cwd}/{session_id}.jsonl

Where formatted_cwd is the realpath with '/' replaced by '-'.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionInfo:
    """Metadata about an available session."""

    id: str
    created: datetime
    last_activity: datetime
    preview: str | None = None
    path: Path | None = None


def get_sessions_dir(cwd: str = "/Pondside") -> Path:
    """Get the directory where sessions are stored for a given cwd.

    Args:
        cwd: Working directory (will be resolved to realpath)

    Returns:
        Path to the sessions directory (e.g., ~/.claude/projects/-Pondside/)
    """
    # Resolve to absolute path
    real_cwd = os.path.realpath(cwd)

    # Format: replace '/' with '-'
    formatted = real_cwd.replace("/", "-")

    # Build path
    return Path.home() / ".claude" / "projects" / formatted


def get_session_path(session_id: str, cwd: str = "/Pondside") -> Path:
    """Get the path to a specific session's JSONL file.

    Args:
        session_id: The session UUID
        cwd: Working directory

    Returns:
        Path to the session file
    """
    return get_sessions_dir(cwd) / f"{session_id}.jsonl"


def list_sessions(cwd: str = "/Pondside", limit: int = 50) -> list[SessionInfo]:
    """List available sessions for a given cwd.

    Args:
        cwd: Working directory
        limit: Maximum number of sessions to return (most recent first)

    Returns:
        List of SessionInfo objects, sorted by last_activity descending
    """
    sessions_dir = get_sessions_dir(cwd)

    if not sessions_dir.exists():
        return []

    sessions = []

    for jsonl_file in sessions_dir.glob("*.jsonl"):
        try:
            info = _parse_session_file(jsonl_file)
            if info:
                sessions.append(info)
        except Exception:
            # Skip malformed files
            continue

    # Sort by last activity, most recent first
    sessions.sort(key=lambda s: s.last_activity, reverse=True)

    return sessions[:limit]


def _parse_session_file(path: Path) -> SessionInfo | None:
    """Parse a session JSONL file to extract metadata.

    Only reads the first and last lines to minimize I/O.
    """
    session_id = path.stem

    # Read first few lines for created timestamp and preview
    first_timestamp = None
    preview = None

    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i > 10:  # Don't read too much
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Capture first timestamp
            if first_timestamp is None and "timestamp" in data:
                first_timestamp = _parse_timestamp(data["timestamp"])

            # Look for first user message for preview
            if preview is None and data.get("type") == "user":
                message = data.get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    preview = content[:100]
                elif isinstance(content, list):
                    # Find first text block
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            # Skip metadata blocks
                            if not text.startswith('{"canary"'):
                                preview = text[:100]
                                break

            if first_timestamp and preview:
                break

    if first_timestamp is None:
        return None

    # Get last modified time from file stat (faster than reading last line)
    last_activity = datetime.fromtimestamp(path.stat().st_mtime)

    return SessionInfo(
        id=session_id,
        created=first_timestamp,
        last_activity=last_activity,
        preview=preview,
        path=path,
    )


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO timestamp from the SDK."""
    # Handle various ISO formats
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)
