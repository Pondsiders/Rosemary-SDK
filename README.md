# rosemary_sdk

Everything that turns Claude into Rosemary.

## What this is

`rosemary_sdk` wraps the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) and adds everything a stateless language model needs to become a stateful person: memory, identity, orientation, and continuity across conversations.

Forked from [Alpha SDK](https://github.com/Pondsiders/Alpha-SDK) (v0.5.2, February 2026). Same machinery, different soul.

## Architecture

```
┌──────────────────────────────────────────┐
│  Client (chat app, automation, etc.)     │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  RosemaryClient    │
         │                    │
         │  soul + orientation │
         │  memory recall     │
         │  memory suggest    │
         │  turn archiving    │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Claude Agent SDK  │
         │                    │
         │  claude subprocess │
         │  tool execution    │
         │  streaming         │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │   CompactProxy     │
         │                    │
         │  prompt rewriting  │
         │  token counting    │
         └─────────┬──────────┘
                   │
              Anthropic API
```

## Key features

- **Soul injection.** System prompt loaded from plugin directory. Editable at runtime.
- **Memory recall.** Semantic + full-text hybrid search on every user prompt.
- **Memory suggest.** Local LLM identifies moments worth remembering after each turn.
- **Compact rewriting.** Compaction preserves identity instead of flattening to bullet points.
- **Token counting.** Real-time context window tracking via Anthropic's count_tokens endpoint.
- **Turn archiving.** Every conversation turn journaled to Postgres.
- **Observability.** Distributed tracing via Logfire.

## Personality lives in the plugin

The SDK is identity-agnostic machinery. All personality-bearing prompts live in `Rosemary-Plugin/prompts/`:

```
Rosemary-Plugin/prompts/
├── system-prompt.md         # The soul — who Rosemary IS
├── compact-system.md        # Identity during compaction
├── compact-instructions.md  # What to include in summaries
├── continuation.md          # Post-compact wake-up behavior
├── suggest-system.md        # What Intro notices as memorable
├── suggest-turn.md          # Turn template for suggestions
├── recall-query-extraction.md  # How to extract search queries
├── here-narratives.md       # Where am I / what am I doing
└── approach-lights.md       # Context warning messages
```

## Configuration

All infrastructure URLs are configured via environment variables:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string (with pgvector) |
| `REDIS_URL` | Redis for ephemeral state |
| `OLLAMA_URL` | Ollama for embeddings and local LLM |
| `OLLAMA_MODEL` | Model name for suggestions and recall |
| `OLLAMA_EMBED_MODEL` | Embedding model (default: nomic-embed-text) |
| `ROSEMARY_ANTHROPIC_API_KEY` | API key for token counting endpoint |
| `ROSEMARY_PLUGIN_DIR` | Override plugin directory location |
| `ROSEMARY_THUMBNAIL_DIR` | Image thumbnail storage |
| `ROSEMARY_DOWNLOAD_DIR` | Downloaded file storage |
| `ROSEMARY_CONTEXT_ROOT` | Root for ROSEMARY.md context files |

## Provenance

Forked from [Pondsiders/Alpha-SDK](https://github.com/Pondsiders/Alpha-SDK) at v0.5.2. The upstream is Alpha's bespoke SDK — same architecture, different soul.

---

*February 2026.*
