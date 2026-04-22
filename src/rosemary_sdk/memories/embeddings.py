"""Embeddings client for Cortex.

Uses nomic-embed-text (or v1.5) with proper task prefixes:
- search_document: for storing memories
- search_query: for searching memories

Talks to whatever OpenAI-compatible endpoint is configured via
OPENAI_BASE_URL (llmster, LM Studio, Harbormaster, OpenAI, etc.).
"""

import os

import logfire
from openai import APIConnectionError, APIError, APITimeoutError

from ..inference_client import get_client

# Configuration from environment — crash at import time if not set
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


async def embed_document(content: str) -> list[float]:
    """Generate embedding for a document (for storage).

    Uses the 'search_document:' task prefix as required by nomic-embed-text.
    """
    return await _embed(f"search_document: {content}")


async def embed_query(query: str) -> list[float]:
    """Generate embedding for a query (for search).

    Uses the 'search_query:' task prefix as required by nomic-embed-text.
    """
    return await _embed(f"search_query: {query}")


async def _embed(prompt: str, timeout: float = 5.0) -> list[float]:
    """Call the OpenAI-compatible endpoint to generate an embedding."""
    with logfire.span(
        "cortex.embed",
        model=EMBED_MODEL,
        prompt_preview=prompt[:50],
    ) as span:
        try:
            response = await get_client().embeddings.create(
                model=EMBED_MODEL,
                input=prompt,
                timeout=timeout,
            )
            embedding = response.data[0].embedding
            span.set_attribute("embedding_dim", len(embedding))
            return embedding
        except APITimeoutError:
            logfire.warning(f"Embedding timeout after {timeout}s")
            raise EmbeddingError("Embedding service timed out")
        except APIConnectionError:
            logfire.warning("Embedding service unreachable")
            raise EmbeddingError("Embedding service unreachable")
        except APIError as e:
            logfire.warning("Embedding API error: {error}", error=str(e))
            raise EmbeddingError(f"Embedding service error: {e}")
        except Exception as e:
            logfire.error(f"Embedding unexpected error: {e}")
            raise EmbeddingError(f"Embedding failed: {e}")


async def health_check() -> bool:
    """Check if the inference endpoint is reachable."""
    try:
        await get_client().models.list()
        return True
    except Exception:
        return False
