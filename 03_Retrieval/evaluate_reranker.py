"""
Cross-Encoder 리랭커 성능 평가 스크립트
 - Baseline: BM25 (Mecab)
 - Rerank: Cross-Encoder 재정렬 후 Top-5
"""

import time
from typing import List, Dict, Tuple

import numpy as np
from mecab import MeCab as MeCabKo

from bm25_retriever import MecabBM25Retriever
from compare_models_dense import load_dataset as load_eval_dataset
from cross_encoder_reranker import CrossEncoderReranker


def find_positive_rank(documents, positive_text: str, tokenizer: MeCabKo, threshold: float = 0.3) -> int:
    """문서 리스트에서 positive 텍스트와 겹치는 문서를 찾아 순위를 반환."""
    positive_tokens = set(tokenizer.morphs(positive_text))
    if not positive_tokens:
        return 999

    for rank, doc in enumerate(documents, start=1):
        doc_tokens = set(tokenizer.morphs(doc.page_content))
        if positive_text in doc.page_content:
            return rank
        overlap = len(positive_tokens & doc_tokens)
        if len(positive_tokens) > 0 and (overlap / len(positive_tokens)) >= threshold:
            return rank
    return 999


def evaluate_reranker(
    retriever: MecabBM25Retriever,
    reranker: CrossEncoderReranker,
    dataset: List[Dict],
    top_n: int = 10,
    top_k: int = 5,
) -> Dict[str, float]:
    tokenizer = MeCabKo()
    retriever.k = top_n

    baseline_ranks = []
    rerank_ranks = []
    fetch_times = []
    rerank_times = []

    for idx, sample in enumerate(dataset, start=1):
        query = sample["query"]
        positive = sample["positive"]

        start = time.time()
        bm25_docs = retriever.invoke(query)
        fetch_times.append(time.time() - start)

        baseline_rank = find_positive_rank(bm25_docs[:top_k], positive, tokenizer)
        baseline_ranks.append(baseline_rank)

        start = time.time()
        reranked_docs, _ = reranker.rerank(query, bm25_docs, top_k=top_k)
        rerank_times.append(time.time() - start)

        rerank_rank = find_positive_rank(reranked_docs, positive, tokenizer)
        rerank_ranks.append(rerank_rank)

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(dataset)} queries...")

    def compute_metrics(ranks):
        recall = np.mean([1 if r <= top_k else 0 for r in ranks])
        mrr = np.mean([1 / r if r <= top_k else 0 for r in ranks])
        return recall, mrr

    baseline_recall, baseline_mrr = compute_metrics(baseline_ranks)
    rerank_recall, rerank_mrr = compute_metrics(rerank_ranks)

    return {
        "baseline_recall": baseline_recall,
        "baseline_mrr": baseline_mrr,
        "rerank_recall": rerank_recall,
        "rerank_mrr": rerank_mrr,
        "avg_fetch_time": float(np.mean(fetch_times)),
        "avg_rerank_time": float(np.mean(rerank_times)),
    }


def main():
    print("Loading dataset (50 samples)...")
    dataset = load_eval_dataset()
    print(f"Loaded {len(dataset)} queries.")

    print("Loading BM25 retriever...")
    retriever = MecabBM25Retriever.load_index(
        "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    )

    print("Loading Cross-Encoder reranker...")
    reranker = CrossEncoderReranker()

    metrics = evaluate_reranker(retriever, reranker, dataset)

    print("\n=== Reranker Evaluation ===")
    print(f"Baseline Recall@5 : {metrics['baseline_recall']:.4f}")
    print(f"Baseline MRR@5    : {metrics['baseline_mrr']:.4f}")
    print(f"Rerank Recall@5   : {metrics['rerank_recall']:.4f}")
    print(f"Rerank MRR@5      : {metrics['rerank_mrr']:.4f}")
    print(f"Avg BM25 Time     : {metrics['avg_fetch_time']:.4f}s")
    print(f"Avg Rerank Time   : {metrics['avg_rerank_time']:.4f}s")


if __name__ == "__main__":
    main()

