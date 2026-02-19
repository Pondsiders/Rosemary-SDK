"""Associative recall - what sounds familiar from this prompt?

Given a user prompt, searches Cortex using two parallel strategies:
1. Direct embedding search (fast, catches overall semantic similarity)
2. OLMo query extraction (slower, catches distinctive terms in long prompts)

Both strategies also search the Sage archive (Kylee's previous AI conversations)
with a flat penalty so own memories always rank above equivalent archive hits.

Results are merged, deduped, sorted by score, and filtered via session-scoped
seen-caches (separate for memories and archive hits).
"""

import asyncio
import json
import os
from typing import Any

import httpx
import logfire

from ..prompts import load_prompt
from .cortex import search as cortex_search, search_sage as sage_search

# Configuration from environment — crash at import time if not set
OLLAMA_URL = os.environ["OLLAMA_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]

# Memory search parameters
DIRECT_LIMIT = 1   # Top 1 own memory for general semantic similarity
QUERY_LIMIT = 1    # Top 1 own memory per extracted query
MIN_SCORE = 0.1    # Minimum similarity threshold

# Sage archive search parameters
SAGE_DIRECT_LIMIT = 2   # Top 2 archive hits for direct search
SAGE_QUERY_LIMIT = 1    # Top 1 archive hit per extracted query
SAGE_KYLEE_PENALTY = 0.15   # Kylee's words — smaller penalty, more likely to surface
SAGE_SAGE_PENALTY = 0.25    # Sage's responses — bigger penalty, more likely generic

# Module-level seen-ID caches, keyed by session_id.
# Separate caches for memories vs archive (different tables, different ID spaces).
_seen_ids: dict[str, set[int]] = {}
_seen_sage_ids: dict[str, set[int]] = {}


def get_seen_ids(session_id: str) -> set[int]:
    """Get the set of memory IDs already seen this session."""
    return _seen_ids.get(session_id, set())


def get_seen_sage_ids(session_id: str) -> set[int]:
    """Get the set of Sage message IDs already seen this session."""
    return _seen_sage_ids.get(session_id, set())


def mark_seen(session_id: str, memory_ids: list[int]) -> None:
    """Mark memory IDs as seen for this session."""
    if not memory_ids:
        return
    if session_id not in _seen_ids:
        _seen_ids[session_id] = set()
    _seen_ids[session_id].update(memory_ids)


def mark_sage_seen(session_id: str, sage_ids: list[int]) -> None:
    """Mark Sage message IDs as seen for this session."""
    if not sage_ids:
        return
    if session_id not in _seen_sage_ids:
        _seen_sage_ids[session_id] = set()
    _seen_sage_ids[session_id].update(sage_ids)


def clear_seen(session_id: str | None = None) -> None:
    """Clear seen IDs for a session (or all sessions if None)."""
    if session_id:
        _seen_ids.pop(session_id, None)
        _seen_sage_ids.pop(session_id, None)
    else:
        _seen_ids.clear()
        _seen_sage_ids.clear()


async def _extract_queries(message: str) -> list[str]:
    """Extract search queries from a user message using Ollama.

    Returns 0-3 descriptive queries, or empty list if message doesn't warrant search.
    """
    template = load_prompt("recall-query-extraction")
    prompt = template.format(message=message[:2000])

    with logfire.span(
        "recall.extract_queries",
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "ollama",
            "gen_ai.request.model": OLLAMA_MODEL,
        }
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                        "options": {"num_ctx": 4096},
                    },
                )
                response.raise_for_status()

            result = response.json()
            output = result.get("message", {}).get("content", "")

            span.set_attribute("gen_ai.usage.input_tokens", result.get("prompt_eval_count", 0))
            span.set_attribute("gen_ai.usage.output_tokens", result.get("eval_count", 0))
            span.set_attribute("gen_ai.response.model", OLLAMA_MODEL)

            parsed = json.loads(output)
            queries = parsed.get("queries", [])

            if isinstance(queries, list):
                valid = [q for q in queries if isinstance(q, str) and q.strip()]
                logfire.debug("Extracted queries", count=len(valid), queries=valid)
                return valid

            return []

        except json.JSONDecodeError as e:
            logfire.warning("Failed to parse OLMo response as JSON", error=str(e))
            return []
        except Exception as e:
            logfire.error("Query extraction failed", error=str(e))
            return []


async def _search_extracted_queries(
    queries: list[str],
    exclude: list[int],
) -> list[dict[str, Any]]:
    """Search Cortex for each extracted query, taking top 1 per query."""
    if not queries:
        return []

    async def search_one(query: str) -> dict[str, Any] | None:
        results = await cortex_search(
            query=query,
            limit=QUERY_LIMIT,
            exclude=exclude,
            min_score=MIN_SCORE,
        )
        return results[0] if results else None

    with logfire.span("recall.search_extracted", query_count=len(queries)) as span:
        tasks = [search_one(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Instrumentation
        query_results = {
            q: (r["id"] if r else None)
            for q, r in zip(queries, results)
        }
        span.set_attribute("query_results", str(query_results))

        # Filter None and dedupe
        memories = []
        seen_in_batch = set(exclude)
        for i, mem in enumerate(results):
            if mem and mem["id"] not in seen_in_batch:
                memories.append(mem)
                seen_in_batch.add(mem["id"])
                logfire.debug(f"Query '{queries[i]}' -> memory #{mem['id']}")
            elif mem:
                logfire.debug(f"Query '{queries[i]}' -> memory #{mem['id']} (deduped)")
            else:
                logfire.debug(f"Query '{queries[i]}' -> no result above threshold")

        return memories


async def _search_sage_extracted_queries(
    queries: list[str],
    exclude: list[int],
) -> list[dict[str, Any]]:
    """Search Sage archive for each extracted query, taking top 1 per query."""
    if not queries:
        return []

    async def search_one(query: str) -> dict[str, Any] | None:
        results = await sage_search(
            query=query,
            limit=SAGE_QUERY_LIMIT,
            kylee_penalty=SAGE_KYLEE_PENALTY,
            sage_penalty=SAGE_SAGE_PENALTY,
            exclude=exclude,
            min_score=MIN_SCORE,
        )
        return results[0] if results else None

    with logfire.span("recall.search_sage_extracted", query_count=len(queries)) as span:
        tasks = [search_one(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Filter None and dedupe
        hits = []
        seen_in_batch = set(exclude)
        for i, hit in enumerate(results):
            if hit and hit["id"] not in seen_in_batch:
                hits.append(hit)
                seen_in_batch.add(hit["id"])
                logfire.debug(f"Sage query '{queries[i]}' -> msg #{hit['id']}")
            elif hit:
                logfire.debug(f"Sage query '{queries[i]}' -> msg #{hit['id']} (deduped)")
            else:
                logfire.debug(f"Sage query '{queries[i]}' -> no result above threshold")

        return hits


async def recall(prompt: str, session_id: str) -> list[dict[str, Any]]:
    """
    Associative recall: what sounds familiar from this prompt?

    Uses three parallel strategies:
    1. Direct embedding search of own memories
    2. Direct embedding search of Sage archive (penalized)
    3. OLMo query extraction + search of both sources

    Results are merged by score (highest first) and deduped.
    Filtered via in-process seen-caches (separate for memories and archive).

    Args:
        prompt: The user's message
        session_id: Current session ID (for seen-cache scoping)

    Returns:
        List of result dicts. Own memories have keys: id, content, created_at, score.
        Archive hits additionally have: speaker, conversation_title, source="archive".
    """
    with logfire.span("recall", session_id=session_id[:8] if session_id else "none") as span:
        seen = get_seen_ids(session_id)
        seen_sage = get_seen_sage_ids(session_id)
        seen_list = list(seen)
        seen_sage_list = list(seen_sage)
        logfire.debug("Seen IDs loaded", memories=len(seen_list), sage=len(seen_sage_list))

        # Phase 1: Run direct search (memories), direct sage search, and
        # query extraction all in parallel
        direct_task = cortex_search(
            query=prompt,
            limit=DIRECT_LIMIT,
            exclude=seen_list if seen_list else None,
            min_score=MIN_SCORE,
        )
        sage_direct_task = sage_search(
            query=prompt,
            limit=SAGE_DIRECT_LIMIT,
            kylee_penalty=SAGE_KYLEE_PENALTY,
            sage_penalty=SAGE_SAGE_PENALTY,
            exclude=seen_sage_list if seen_sage_list else None,
            min_score=MIN_SCORE,
        )
        extract_task = _extract_queries(prompt)

        direct_memories, sage_direct, extracted_queries = await asyncio.gather(
            direct_task, sage_direct_task, extract_task
        )

        span.set_attribute("extracted_queries", extracted_queries)
        span.set_attribute("direct_memory_ids", [m["id"] for m in direct_memories])
        span.set_attribute("sage_direct_ids", [s["id"] for s in sage_direct])

        # Phase 2: Search extracted queries against both sources in parallel
        # Build exclude lists to avoid dupes
        exclude_for_extracted = set(seen_list)
        for mem in direct_memories:
            exclude_for_extracted.add(mem["id"])

        exclude_sage_for_extracted = set(seen_sage_list)
        for hit in sage_direct:
            exclude_sage_for_extracted.add(hit["id"])

        extracted_memories_task = _search_extracted_queries(
            extracted_queries,
            list(exclude_for_extracted),
        )
        sage_extracted_task = _search_sage_extracted_queries(
            extracted_queries,
            list(exclude_sage_for_extracted),
        )

        extracted_memories, sage_extracted = await asyncio.gather(
            extracted_memories_task, sage_extracted_task
        )

        span.set_attribute("extracted_memory_ids", [m["id"] for m in extracted_memories])
        span.set_attribute("sage_extracted_ids", [s["id"] for s in sage_extracted])

        # Phase 3: Merge all results by score (highest first)
        # Tag memory results with source
        all_results = []
        for mem in direct_memories + extracted_memories:
            mem["source"] = "memory"
            all_results.append(mem)

        # Sage results already have source="archive" from db.py
        all_results.extend(sage_direct + sage_extracted)

        # Sort by score descending
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        span.set_attribute("total_results", len(all_results))

        if not all_results:
            logfire.info("No results above threshold")
            return []

        # Mark as seen (separate caches)
        new_memory_ids = [r["id"] for r in all_results if r.get("source") == "memory"]
        new_sage_ids = [r["id"] for r in all_results if r.get("source") == "archive"]
        mark_seen(session_id, new_memory_ids)
        mark_sage_seen(session_id, new_sage_ids)

        logfire.debug(
            "Recall complete",
            memories=len(direct_memories) + len(extracted_memories),
            archive=len(sage_direct) + len(sage_extracted),
            total=len(all_results),
        )

        return all_results
