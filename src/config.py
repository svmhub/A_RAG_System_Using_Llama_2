"""
config.py
---------
Central place for every tunable globel configuration setting in the project.

WHY THIS FILE EXISTS (viva point):
Hard coding values like chunk size or model names inside every module makes
the codebase brittle -- change one thing, hunt through five files. Keeping
all the variables in one config module is a standard software-engineering practice
(separation of configuration from logic) and makes the system easy to
re-tune during evaluation (Step 5 of the project) without touching pipeline
code.
"""

import os

# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------
# CHUNK_SIZE: number of characters per text chunk stored in the vector DB.
# CHUNK_OVERLAP: characters shared between consecutive chunks so a sentence
# that straddles a chunk boundary isn't cut in half and loses meaning.

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2 is a SentenceTransformers model: 384-dimensional
# embeddings, ~80MB, runs fast on CPU. Good enough for retrieval quality
# while being practical for a student project without a GPU.

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = os.path.join("vector_store", "faiss_index.bin")
METADATA_PATH = os.path.join("vector_store", "chunks_metadata.pkl")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K = 4  # how many chunks to retrieve per query

# ---------------------------------------------------------------------------
# Generation model (LLM)
# ---------------------------------------------------------------------------
# True Llama-2-7b-chat requires a Hugging Face token with access approved by
# Meta, and ~13GB VRAM in fp16 (or ~6GB with 4-bit quantization).

LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# Fallback model: no gating, no login needed, small enough to run on CPU.
# Used automatically if Llama-2 can't be loaded (no token / no GPU), so the
# app is still demoable. Same RAG pipeline, swap-in generator only.

FALLBACK_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 400
TEMPERATURE = 0.3  # low temperature -> more grounded, less "creative" answers

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# This is the core of "prompt engineering" mentioned in Step 4 of the brief.
# It instructs the model to answer ONLY from the retrieved context and to
# admit when it doesn't know, which is what makes this a RAG system rather
# than a standalone LLM prone to hallucination.

LLAMA2_CHAT_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a precise, factual assistant. Answer the user's question using ONLY
the context provided below. If the answer is not contained in the context,
say "I don't have enough information in the provided documents to answer
that." Do not use outside knowledge. Cite which part of the context you used
where relevant.
<</SYS>>

Context:
{context}

Question: {question} [/INST]"""
