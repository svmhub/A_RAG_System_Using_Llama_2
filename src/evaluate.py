"""
evaluate.py
-----------
STEP 5 of the pipeline: Evaluation & Optimization.

Compares generated answers against ground-truth answers using three
complementary metric families, each catching something the others miss:

  1. BLEU     - precision-based n-gram overlap. Penalizes answers that add
                extra words not in the reference. Originally built for
                machine translation.
  2. ROUGE-L  - recall-based, uses the Longest Common Subsequence. Rewards
                answers that cover the reference's content even if worded
                differently. Originally built for summarization.
  3. Embedding cosine similarity - the only metric here that captures
                MEANING rather than exact word overlap. Two answers can be
                worded completely differently and still score highly if
                they mean the same thing. This is important for LLM
                outputs, which rarely match a reference string exactly.

WHY USE ALL THREE (viva point):
BLEU/ROUGE are lexical (word-overlap) metrics -- fast and standard, but
they penalize a *correct* answer that's just phrased differently from the
reference. Embedding similarity fixes that blind spot, at the cost of not
telling you anything about fluency or exact factual phrasing. Reporting
all three gives a fuller picture than trusting any single number.
"""

from typing import List, Dict

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .embed_index import VectorStore


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_embedding_similarity(reference: str, hypothesis: str,
                                  vector_store: VectorStore) -> float:
    vecs = vector_store.embed([reference, hypothesis])
    # Vectors are already L2-normalized (see embed_index.py), so the dot
    # product IS the cosine similarity.
    return float(np.dot(vecs[0], vecs[1]))


def evaluate_answer(reference: str, hypothesis: str,
                     vector_store: VectorStore) -> Dict[str, float]:
    return {
        "bleu": compute_bleu(reference, hypothesis),
        "rouge_l": compute_rouge_l(reference, hypothesis),
        "embedding_similarity": compute_embedding_similarity(
            reference, hypothesis, vector_store
        ),
    }


def evaluate_batch(pairs: List[Dict[str, str]],
                    vector_store: VectorStore) -> List[Dict]:
    """
    pairs: list of {"question": ..., "reference": ..., "hypothesis": ...}
    Returns per-example scores plus an averaged summary row.
    """
    results = []
    for pair in pairs:
        scores = evaluate_answer(pair["reference"], pair["hypothesis"], vector_store)
        results.append({**pair, **scores})

    if results:
        avg = {
            "question": "AVERAGE",
            "bleu": float(np.mean([r["bleu"] for r in results])),
            "rouge_l": float(np.mean([r["rouge_l"] for r in results])),
            "embedding_similarity": float(np.mean([r["embedding_similarity"] for r in results])),
        }
        results.append(avg)

    return results
