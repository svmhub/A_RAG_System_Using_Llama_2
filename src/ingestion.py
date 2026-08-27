"""
ingestion.py
------------
STEP 1 of the pipeline: Data Ingestion.

Responsibilities:
  1. Accept PDF, DOCX, or TXT files.
  2. Extract raw text from each format using the right library.
  3. Split ("chunk") the text into overlapping windows so it can be
     embedded and retrieved at a useful granularity.

WHY WE CHUNK (viva point):
LLMs and embedding models have limited context windows, and embedding an
entire 50-page PDF as ONE vector would blur together many unrelated topics,
making retrieval useless (a query about "page 40" would score similarly to
one about "page 2"). Chunking into ~1000-character windows keeps each
vector semantically focused on one idea, which is what makes similarity
search (Step 3) actually work. The overlap (150 chars) prevents losing
context when a sentence or idea is split exactly at a chunk boundary.
"""

from dataclasses import dataclass
from typing import List
import os

import fitz  # PyMuPDF
import docx2txt

from . import config


@dataclass
class Chunk:
    """A single unit of retrievable text plus where it came from."""
    text: str
    source: str      # original filename
    chunk_id: int     # position within that file


# ---------------------------------------------------------------------------
# Text extraction (format-specific)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> str:
    """
    Uses PyMuPDF (imported as `fitz`) to pull text page by page.
    PyMuPDF is chosen over PDFPlumber here because it's noticeably faster on
    large PDFs and handles most layouts well; PDFPlumber is better if you
    specifically need table extraction, which is why the brief lists both
    as options.
    """
    text_parts = []
    with fitz.open(filepath) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(filepath: str) -> str:
    """docx2txt handles paragraphs and (optionally) embedded images/text."""
    return docx2txt.process(filepath)


def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(filepath: str) -> str:
    """Dispatches to the correct extractor based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str,
               chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[Chunk]:
    """
    Simple sliding-window chunker over raw characters.

    A more advanced version (mention in viva as an improvement) would chunk
    on sentence/paragraph boundaries using something like LangChain's
    RecursiveCharacterTextSplitter, so chunks don't cut words/sentences in
    half. This manual version is kept dependency-light and transparent for
    demonstrating the concept.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0
    step = max(chunk_size - overlap, 1)

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_id=chunk_id))
            chunk_id += 1
        start += step

    return chunks


def process_documents(filepaths: List[str]) -> List[Chunk]:
    """End-to-end ingestion: extract + chunk every uploaded file."""
    all_chunks: List[Chunk] = []
    for path in filepaths:
        filename = os.path.basename(path)
        try:
            raw_text = extract_text(path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {filename}: {e}")
        all_chunks.extend(chunk_text(raw_text, source=filename))
    return all_chunks
