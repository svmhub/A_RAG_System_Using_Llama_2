"""
app.py
------
Streamlit frontend. This file wires together every step of the pipeline:

  Upload (Step 1) -> Embed & Index (Step 2) -> Ask a question -> Retrieve
  (Step 3) -> Generate (Step 4) -> optionally Evaluate (Step 5)

WHY STREAMLIT (viva point):
Streamlit turns a plain Python script into a web UI with no separate
frontend code (no HTML/CSS/JS, no Flask/FastAPI routes to write by hand).
Every widget (st.file_uploader, st.button, st.chat_input) is one line and
the whole script just re-runs top-to-bottom on every interaction, with
st.session_state used to persist things (like the vector index) across
those re-runs. That simplicity is exactly why it's a common choice for
ML/AI demo apps and student projects.
"""

import os
import tempfile

import streamlit as st

from src import config
from src.ingestion import process_documents
from src.embed_index import VectorStore
from src.retrieval import retrieve_context
from src.generation import Generator
from src.evaluate import evaluate_answer

st.set_page_config(page_title="RAG Document Q&A (Llama 2)", layout="wide")
st.title("📄 RAG Document Q&A — powered by Llama 2")
st.caption(
    "Upload PDFs, DOCX, or TXT files, then ask questions. Answers are "
    "generated only from the content you upload."
)

# -----------------------------------------------------------------------
# Session state: Streamlit re-runs this whole script on every interaction,
# so anything that shouldn't be rebuilt from scratch (the vector index, the
# loaded LLM, chat history) must live in st.session_state.
# -----------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "generator" not in st.session_state:
    st.session_state.generator = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

# -----------------------------------------------------------------------
# Sidebar: document upload + indexing (Steps 1 & 2)
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    if st.button("Process & Index Documents", disabled=not uploaded_files):
        with st.spinner("Extracting text and building embeddings..."):
            tmp_paths = []
            tmp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                tmp_paths.append(path)

            chunks = process_documents(tmp_paths)
            st.session_state.vector_store.build_index(chunks)
            st.session_state.indexed = True

        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    st.header("2. Load the LLM")
    hf_token = st.text_input(
        "Hugging Face token (only needed for gated Llama-2 access)",
        type="password",
    )
    use_4bit = st.checkbox("Use 4-bit quantization (recommended on GPU)", value=True)

    if st.button("Load Llama 2 (or fallback)"):
        with st.spinner("Loading model... this can take a few minutes the first time."):
            st.session_state.generator = Generator(use_4bit=use_4bit, hf_token=hf_token or None)
        st.success(f"Loaded: {st.session_state.generator.model_name_loaded}")

    st.divider()
    st.header("3. Retrieval settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, config.TOP_K)

# -----------------------------------------------------------------------
# Main panel: chat interface (Steps 3 & 4)
# -----------------------------------------------------------------------
if not st.session_state.indexed:
    st.info("Upload and index documents in the sidebar to get started.")
elif st.session_state.generator is None:
    st.info("Load the LLM in the sidebar to start asking questions.")
else:
    question = st.chat_input("Ask a question about your documents...")

    if question:
        with st.spinner("Retrieving relevant context..."):
            context, results = retrieve_context(
                st.session_state.vector_store, question, top_k=top_k
            )
        with st.spinner("Generating answer..."):
            answer = st.session_state.generator.generate(question, context)

        st.session_state.chat_history.append((question, answer, results))

    for q, a, sources in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if sources:
                with st.expander("Show retrieved sources"):
                    for chunk, score in sources:
                        st.markdown(
                            f"**{chunk.source}** (chunk #{chunk.chunk_id}, "
                            f"similarity={score:.3f})"
                        )
                        st.text(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))

# -----------------------------------------------------------------------
# Optional: evaluation panel (Step 5)
# -----------------------------------------------------------------------
with st.expander("📊 Evaluate an answer against a ground-truth reference"):
    st.write(
        "Paste a reference (ground-truth) answer to score the most recent "
        "generated answer using BLEU, ROUGE-L, and embedding similarity."
    )
    reference = st.text_area("Ground-truth answer")
    if st.button("Evaluate last answer") and st.session_state.chat_history:
        _, last_answer, _ = st.session_state.chat_history[-1]
        scores = evaluate_answer(reference, last_answer, st.session_state.vector_store)
        col1, col2, col3 = st.columns(3)
        col1.metric("BLEU", f"{scores['bleu']:.3f}")
        col2.metric("ROUGE-L", f"{scores['rouge_l']:.3f}")
        col3.metric("Embedding similarity", f"{scores['embedding_similarity']:.3f}")
