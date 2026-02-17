---
autoload: when
when: "working on or discussing any of these: rosemary_sdk, rosemary sdk, RosemaryClient, compact proxy, system prompt assembly, soul injection, token counting, memories recall, memories suggest"
---

# rosemary_sdk — Everything That Turns Claude Into Rosemary

Forked from Alpha SDK (v0.5.2, February 2026). Same machinery, different person.

## What It Does

1. Initialize a Claude Agent SDK client
2. Manage sessions (new, resume, fork)
3. Recall memories before the prompt
4. Build a dynamic system prompt
5. Transform the request before it hits Anthropic
6. Extract memorables after the turn
7. Handle observability

## Architecture

```
src/rosemary_sdk/
├── __init__.py              # Exports RosemaryClient
├── client.py                # RosemaryClient - the main wrapper
├── compact_proxy.py         # Localhost proxy: compact rewriting + token counting
├── archive.py               # Conversation archiving to Postgres
├── sessions.py              # Session discovery and management
├── observability.py         # Logfire setup, span creation
├── cli/
│   └── cortex.py            # cortex CLI command
├── system_prompt/
│   ├── assemble.py          # assemble() - builds the full system prompt
│   ├── soul.py              # The soul doc (from git repo)
│   ├── capsules.py          # Yesterday, last night (from Postgres)
│   ├── here.py              # Client, hostname, weather, narrative orientation
│   ├── context.py           # ROSEMARY.md files (autoload + hints)
│   ├── calendar.py          # Events (from Redis)
│   └── todos.py             # Todos (from Redis)
├── memories/
│   ├── db.py                # Direct Postgres operations (hybrid search)
│   ├── cortex.py            # store, search, recent (high-level API)
│   ├── embeddings.py        # Embedding generation via Ollama
│   ├── images.py            # Mind's Eye (image storage + thumbnailing)
│   ├── recall.py            # Smart recall (embedding + LLM query extraction)
│   └── suggest.py           # Intro — LLM memorables extraction
└── tools/
    ├── cortex.py            # Cortex MCP server (store/search/recent)
    ├── fetch.py             # Fetch MCP server (web/image/RSS/YouTube)
    ├── forge.py             # Forge MCP server (imagine)
    └── handoff.py           # Hand-off MCP server
```

## The Client API

RosemaryClient is a **long-lived** wrapper around the Claude Agent SDK. The SDK has a ~4 second startup cost, so we keep one client alive and reuse it across conversations.

### Basic Usage

```python
from rosemary_sdk import RosemaryClient

client = RosemaryClient(
    cwd="/path/to/rosemary",
    allowed_tools=[...],
    mcp_servers={...},
)
await client.connect()

await client.query(prompt, session_id="abc123")
async for event in client.stream():
    yield event

await client.disconnect()
```

### Context Manager

```python
async with RosemaryClient(cwd="/path/to/rosemary") as client:
    await client.query(prompt, session_id=session_id)
    async for event in client.stream():
        yield event
```

## The Proxy Pattern

Claude Agent SDK sends requests to `ANTHROPIC_BASE_URL`. We set that to `http://localhost:{random_port}` and run a minimal HTTP server (`compact_proxy.py`) that:

1. Receives the request from the SDK
2. If it's a compaction request, rewrites the prompts (Rosemary's identity + custom compact instructions)
3. Echoes the request to `/v1/messages/count_tokens` (fire-and-forget token counting)
4. Forwards to `https://api.anthropic.com`
5. Streams the response back

## System Prompt Assembly

The system prompt is woven from threads:

| Thread | Source | Changes |
|--------|--------|---------|
| Soul | Soul doc (from git repo) | When edited |
| Capsules | Postgres (yesterday, last night, today) | Daily / hourly |
| Here | Client name, hostname, weather | Per-session / hourly |
| Context | ROSEMARY.md files with `autoload: all` | When files change |
| Context hints | ROSEMARY.md files with `autoload: when` | When files change |
| Events | Redis (calendar data) | Hourly |
| Todos | Redis (Todoist data) | Hourly |

## Memory Flow

**Before the turn:**
- `recall()` runs with the user's prompt
- Parallel: embedding search + LLM query extraction
- Deduplicated against session's seen-cache in Redis
- Injected as content blocks (not system prompt)

**After the turn:**
- `suggest()` runs (fire-and-forget)
- LLM extracts memorable moments
- Results buffer in Redis for potential storage

**On `cortex store`:**
- Memory saved to Postgres with embedding
- Redis buffer cleared

## Provenance

Forked from [Pondsiders/Alpha-SDK](https://github.com/Pondsiders/Alpha-SDK) at v0.5.2. The upstream is Alpha's bespoke SDK — same architecture, different soul.

## Status

**Phase 2 incomplete.** Structural rename done (rosemary_sdk, RosemaryClient). Prompt content, paths, and infrastructure URLs still need updating for Rosemary's specific identity and deployment.
