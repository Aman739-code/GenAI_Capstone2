"""
RAG Retriever Interface.
Provides a clean interface for querying the FAISS vector store
and retrieving relevant grid guidelines.
"""

from __future__ import annotations

from typing import Optional
import logging

from config.settings import TOP_K_RETRIEVAL
from rag.ingest import load_vectorstore

logger = logging.getLogger(__name__)

# Module-level cache for the vector store
_vectorstore = None


def _get_vectorstore():
    """Lazy-load and cache the vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def retrieve(query: str, k: Optional[int] = None) -> list[dict]:
    """
    Retrieve relevant documents from the FAISS vector store.

    Args:
        query: Semantic search query.
        k: Number of results to return. Defaults to TOP_K_RETRIEVAL.

    Returns:
        List of dicts with keys: 'content', 'source', 'score'.
    """
    k = k or TOP_K_RETRIEVAL

    try:
        vs = _get_vectorstore()
        results = vs.similarity_search_with_score(query, k=k)

        retrieved = []
        for doc, score in results:
            source_path = doc.metadata.get("source", "Unknown")
            # Extract just the filename for cleaner references
            source_name = source_path.split("/")[-1] if "/" in source_path else source_path

            retrieved.append({
                "content": doc.page_content,
                "source": source_name,
                "score": round(float(score), 4),
                "metadata": doc.metadata,
            })

        logger.info(f"Retrieved {len(retrieved)} documents for query: '{query[:80]}...'")
        return retrieved

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return [{
            "content": "Retrieval failed. Using fallback general guidelines: "
                       "Maintain grid frequency within ±0.5 Hz. Keep battery SoC between 20-80%. "
                       "Follow IEEE 1547 interconnection standards.",
            "source": "fallback_guidelines",
            "score": 0.0,
            "metadata": {"error": str(e)},
        }]


def retrieve_multi_query(queries: list[str], k: Optional[int] = None) -> list[dict]:
    """
    Retrieve documents using multiple queries and deduplicate results.

    Args:
        queries: List of semantic search queries.
        k: Number of results per query.

    Returns:
        Deduplicated list of retrieved documents, sorted by relevance score.
    """
    k = k or TOP_K_RETRIEVAL
    all_results = {}

    for query in queries:
        results = retrieve(query, k=k)
        for r in results:
            key = r["content"][:100]  # Use first 100 chars as dedup key
            if key not in all_results or r["score"] < all_results[key]["score"]:
                all_results[key] = r

    # Sort by score (lower = more relevant for FAISS L2 distance)
    sorted_results = sorted(all_results.values(), key=lambda x: x["score"])
    logger.info(f"Multi-query retrieval: {len(queries)} queries → {len(sorted_results)} unique results")
    return sorted_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = retrieve("What are the battery storage charging strategies during peak solar?")
    for r in results:
        print(f"\n[Score: {r['score']:.4f}] Source: {r['source']}")
        print(f"  {r['content'][:200]}...")
