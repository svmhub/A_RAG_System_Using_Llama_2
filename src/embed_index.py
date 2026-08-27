"""
embed_index.py
--------------
STEP 2 of the pipeline: Text Embeddings & Indexing.

Responsibilities:
  1. Convert each text chunk into a dense vector ("embedding") using a
     SentenceTransformers model.
  2. Store those vectors in a FAISS index for fast similarity search.
  3. Persist the index + chunk metadata to disk so the app doesn't need to
     re-embed documents every time it restarts.

WHY FAISS (viva point):
FAISS (Facebook AI Similarity Search) is an in-memory / on-disk vector
index library that does approximate nearest-neighbour search extremely
fast, even over millions of vectors. It's free, runs locally (no external
service/account like Pinecone needs), and is the easiest of the three
options in the brief (FAISS / ChromaDB / Pinecone) to set up for a
self-contained student project. ChromaDB would be a good alternative if you
wanted a persistent DB with built-in metadata filtering; Pinecone is a
managed cloud service better suited for production/multi-user scale.

WHY COSINE SIMILARITY (viva point):
Embeddings encode MEANING, not magnitude. Two chunks that discuss the same
concept should point in a similar DIRECTION in vector space, regardless of
their length/magnitude. Cosine similarity measures the angle between
vectors, which is why it's the standard metric for semantic search
(implemented here via FAISS's inner-product index on L2-normalized
vectors, which is mathematically equivalent to cosine similarity).
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from . import config
from .ingestion import Chunk


class VectorStore:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[Chunk] = []

    # -- Embedding ----------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of strings into an (N, EMBEDDING_DIM) numpy array.
        normalize_embeddings=True makes each vector unit-length, which is
        what turns FAISS's inner-product search into cosine similarity
        search (see module docstring).
        """
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    # -- Index building -------------------------------------------------------

    def build_index(self, chunks: List[Chunk]) -> None:
        """Embeds every chunk and builds a fresh FAISS index in memory."""
        self.chunks = chunks
        if not chunks:
            self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
            return

        vectors = self.embed([c.text for c in chunks])
        # IndexFlatIP = exact (brute-force) inner-product search.
        # Fine for a few thousand chunks (typical student-project scale).
        # For millions of vectors you'd swap in IndexIVFFlat/IndexHNSW for
        # approximate-but-much-faster search.
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self.index.add(vectors)

    def add_documents(self, chunks: List[Chunk]) -> None:
        """Adds new chunks to an existing index (incremental ingestion)."""
        if self.index is None:
            self.build_index(chunks)
            return
        vectors = self.embed([c.text for c in chunks])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    # -- Search ---------------------------------------------------------------

    def search(self, query: str, top_k: int = config.TOP_K) -> List[Tuple[Chunk, float]]:
        """Returns the top_k (chunk, similarity_score) pairs for a query."""
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self.embed([query])
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # -- Persistence ------------------------------------------------------------

    def save(self, index_path: str = config.FAISS_INDEX_PATH,
              metadata_path: str = config.METADATA_PATH) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, index_path: str = config.FAISS_INDEX_PATH,
              metadata_path: str = config.METADATA_PATH) -> None:
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.chunks = pickle.load(f)
