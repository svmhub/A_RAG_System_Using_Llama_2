"""
retrieval.py
------------
STEP 3 of the pipeline: Query Processing & Retrieval.

This module is intentionally thin: the heavy lifting (embedding + cosine
similarity search) already lives in embed_index.VectorStore. This file's
job is to shape the retrieved chunks into a clean context string that's
ready to hand to the LLM in Step 4.

WHY SEPARATE THIS FROM embed_index.py (viva point):
Keeping "how we search" (embed_index.py) separate from "how we use search
results" (retrieval.py) follows the single-responsibility principle. It
also makes it easy to later add re-ranking, filtering by source document,
or deduplication logic here without touching the vector store internals.
"""

from typing import List, Tuple

from .embed_index import VectorStore
from .ingestion import Chunk
from . import config


def retrieve_context(vector_store: VectorStore, query: str,
                      top_k: int = config.TOP_K) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Runs similarity search and formats the results into a single context
    block, with each chunk labelled by its source document so the model
    (and the user, via citations) can trace answers back to a file.
    """
    results = vector_store.search(query, top_k=top_k)

    if not results:
        return "", []

    formatted_pieces = []
    for i, (chunk, score) in enumerate(results, start=1):
        formatted_pieces.append(
            f"[Source {i}: {chunk.source}, chunk #{chunk.chunk_id} | "
            f"similarity={score:.3f}]\n{chunk.text}"
        )
    context = "\n\n".join(formatted_pieces)
    return context, results
