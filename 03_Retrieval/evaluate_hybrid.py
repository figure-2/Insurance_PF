"""
하이브리드 리트리버 효용성 검증
Dense Only vs Sparse Only vs Hybrid 성능 비교
"""

import time
import numpy as np
from typing import List, Dict
from hybrid_retriever import get_dense_retriever, get_sparse_retriever, get_hybrid_retriever
from compare_models_dense import load_dataset as load_eval_dataset


def calculate_recall_at_k(results: List[int], k: int = 5) -> float:
    """Recall@K 계산"""
    if not results:
        return 0.0
    return sum(1 for rank in results if rank <= k) / len(results)


def calculate_mrr_at_k(results: List[int], k: int = 5) -> float:
    """MRR@K 계산"""
    if not results:
        return 0.0
    reciprocal_ranks = []
    for rank in results:
        if rank <= k:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def find_positive_rank(
    query: str,
    positive: str,
    results,
    tokenizer
) -> int:
    """
    검색 결과에서 positive passage가 포함된 문서의 순위 찾기
    
    Returns:
        순위 (1-based), 없으면 999
    """
    positive_tokens = set(tokenizer.morphs(positive))
    
    for rank, item in enumerate(results, start=1):
        doc = item[0] if isinstance(item, (list, tuple)) else item
        doc_text = doc.page_content
        doc_tokens = set(tokenizer.morphs(doc_text))
        
        # 키워드 겹침 비율 계산
        overlap = len(positive_tokens & doc_tokens)
        if overlap >= len(positive_tokens) * 0.3:  # 30% 이상 겹치면 정답으로 간주
            return rank
    
    return 999


def evaluate_retriever(
    retriever,
    eval_dataset: List[Dict],
    retriever_name: str,
    tokenizer
):
    """리트리버 성능 평가"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {retriever_name}")
    print(f"{'='*60}")
    
    ranks = []
    search_times = []
    
    for i, item in enumerate(eval_dataset):
        query = item['query']
        positive = item['positive']
        
        start_time = time.time()
        results = retriever.invoke(query)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        rank = find_positive_rank(query, positive, results, tokenizer)
        
        ranks.append(rank)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(eval_dataset)} queries...")
    
    recall_at_5 = calculate_recall_at_k(ranks, k=5)
    mrr_at_5 = calculate_mrr_at_k(ranks, k=5)
    avg_search_time = np.mean(search_times)
    
    return {
        'name': retriever_name,
        'recall_at_5': recall_at_5,
        'mrr_at_5': mrr_at_5,
        'avg_search_time': avg_search_time,
        'ranks': ranks
    }


def main():
    print("Loading evaluation dataset (50 samples)...")
    eval_dataset = load_eval_dataset()
    print(f"Loaded {len(eval_dataset)} test queries")
    
    # Mecab 토크나이저 (정답 찾기용)
    from mecab import MeCab as MeCabKo
    tokenizer = MeCabKo()
    
    # 리트리버 생성
    print("\nInitializing retrievers...")
    dense_retriever = get_dense_retriever(k=50)
    sparse_retriever = get_sparse_retriever(k=50)
    hybrid_retriever = get_hybrid_retriever(k=5)
    
    # 각 리트리버 평가
    results = []
    
    # 1. Dense Only
    dense_result = evaluate_retriever(
        dense_retriever,
        eval_dataset,
        "Dense Only (ko-sroberta)",
        tokenizer=tokenizer
    )
    results.append(dense_result)
    
    # 2. Sparse Only
    sparse_result = evaluate_retriever(
        sparse_retriever,
        eval_dataset,
        "Sparse Only (Mecab + BM25)",
        tokenizer=tokenizer
    )
    results.append(sparse_result)
    
    # 3. Hybrid
    hybrid_result = evaluate_retriever(
        hybrid_retriever,
        eval_dataset,
        "Hybrid (Dense + Sparse)",
        tokenizer=tokenizer
    )
    results.append(hybrid_result)
    
    # 결과 출력
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Retriever':<25} | {'Recall@5':<10} | {'MRR@5':<10} | {'Avg Time':<10}")
    print("-" * 60)
    
    for res in results:
        print(f"{res['name']:<25} | {res['recall_at_5']:.4f}     | "
              f"{res['mrr_at_5']:.4f}     | {res['avg_search_time']:.4f}s")
    
    # 최고 성능 리트리버 선정
    best_recall = max(results, key=lambda x: x['recall_at_5'])
    best_mrr = max(results, key=lambda x: x['mrr_at_5'])
    
    print("\n" + "="*60)
    print(f"🏆 Best by Recall@5: {best_recall['name']} ({best_recall['recall_at_5']:.4f})")
    print(f"🏆 Best by MRR@5: {best_mrr['name']} ({best_mrr['mrr_at_5']:.4f})")
    print("="*60)
    
    # Win/Loss 분석
    print("\nWin/Loss Analysis (Hybrid vs Dense):")
    hybrid_ranks = hybrid_result['ranks']
    dense_ranks = dense_result['ranks']
    
    wins = sum(1 for h, d in zip(hybrid_ranks, dense_ranks) if h < d)
    losses = sum(1 for h, d in zip(hybrid_ranks, dense_ranks) if h > d)
    ties = sum(1 for h, d in zip(hybrid_ranks, dense_ranks) if h == d)
    
    print(f"  Wins: {wins} (Hybrid이 더 좋은 순위)")
    print(f"  Losses: {losses} (Dense가 더 좋은 순위)")
    print(f"  Ties: {ties} (동일)")


if __name__ == "__main__":
    main()

