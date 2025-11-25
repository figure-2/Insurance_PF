"""
Dense 리트리버 성능 평가 스크립트
 - 데이터셋: 기존 evaluation_dataset(30) + 추가 20개 쿼리
 - 지표: Recall@5, MRR@5, Avg Search Time
"""

import json
import os
import time
from typing import List, Dict

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATASET_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/evaluation_dataset.json"
CHROMA_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/chroma_db"
MODEL_NAME = "jhgan/ko-sroberta-multitask"

from compare_models_dense import load_dataset as load_compare_dataset  # reuse definitions


def find_rank(results, positive_text: str, tokenizer, threshold: float = 0.3) -> int:
    """검색 결과에서 positive_text와 겹치는 문서의 순위를 찾는다."""
    positive_tokens = set(tokenizer.morphs(positive_text))
    if not positive_tokens:
        return 999

    for rank, (doc, _) in enumerate(results, start=1):
        doc_tokens = set(tokenizer.morphs(doc.page_content))
        if positive_text in doc.page_content:
            return rank
        overlap = len(positive_tokens & doc_tokens)
        if len(positive_tokens) == 0:
            continue
        if (overlap / len(positive_tokens)) >= threshold:
            return rank
    return 999


def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma(
        collection_name="insurance_policies",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store


def main():
    dataset = load_compare_dataset()
    vector_store = load_retriever()
    from mecab import MeCab as MeCabKo
    tokenizer = MeCabKo()

    ranks = []
    search_times = []

    for idx, sample in enumerate(dataset, start=1):
        query = sample["query"]
        positive = sample["positive"]

        start = time.time()
        results = vector_store.similarity_search_with_score(query, k=5)
        search_times.append(time.time() - start)

        rank = find_rank(results, positive, tokenizer)
        ranks.append(rank)

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(dataset)} queries...")

    recall = np.mean([1 if r <= 5 else 0 for r in ranks])
    mrr = np.mean([1 / r if r <= 5 else 0 for r in ranks])
    avg_time = np.mean(search_times)

    print("\n=== Dense Retriever Evaluation ===")
    print(f"Total Queries : {len(dataset)}")
    print(f"Recall@5      : {recall:.4f}")
    print(f"MRR@5         : {mrr:.4f}")
    print(f"Avg Time/query: {avg_time:.4f} s")


if __name__ == "__main__":
    main()

