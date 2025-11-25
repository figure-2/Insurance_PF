"""
형태소 분석기 비교 실험: Kiwi vs Mecab vs Okt
BM25 검색 성능을 기준으로 최적 토크나이저 선정
"""

import json
import time
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

# 형태소 분석기 import
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("Warning: Kiwi not available")

try:
    from mecab import MeCab as MeCabKo
    MECAB_KO_AVAILABLE = True
except ImportError:
    MECAB_KO_AVAILABLE = False
    print("Warning: python-mecab-ko not available")

try:
    from konlpy.tag import Okt
    KONLPY_OKT_AVAILABLE = True
except ImportError:
    KONLPY_OKT_AVAILABLE = False
    print("Warning: Konlpy Okt not available")


class TokenizerWrapper:
    """형태소 분석기 래퍼 클래스"""
    
    def __init__(self, name: str, tokenizer):
        self.name = name
        self.tokenizer = tokenizer
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰 리스트로 변환"""
        if self.name == "Kiwi":
            # Kiwi는 analyze() 메서드 사용 (결과: [(형태소, 품사, 시작위치, 끝위치), ...])
            try:
                result = self.tokenizer.analyze(text)
                if result and len(result) > 0 and len(result[0]) > 0:
                    return [morph for morph, pos, _, _ in result[0][0]]
                return []
            except:
                return []
        elif self.name == "Mecab":
            # python-mecab-ko는 morphs() 메서드 사용
            return self.tokenizer.morphs(text)
        elif self.name == "Okt":
            # Okt는 morphs() 메서드 사용
            return self.tokenizer.morphs(text)
        else:
            raise ValueError(f"Unknown tokenizer: {self.name}")


def load_documents(jsonl_path: str) -> List[Dict]:
    """chunked_data.jsonl 로드"""
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            documents.append({
                'chunk_id': item['chunk_id'],
                'text': item['text'],
                'metadata': item['metadata']
            })
    return documents


def load_evaluation_dataset(json_path: str) -> List[Dict]:
    """evaluation_dataset.json 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_recall_at_k(results: List[int], k: int = 5) -> float:
    """Recall@K 계산: 정답이 상위 K개 안에 포함되는지"""
    if not results:
        return 0.0
    return sum(1 for rank in results if rank <= k) / len(results)


def calculate_mrr_at_k(results: List[int], k: int = 5) -> float:
    """MRR@K 계산: 정답의 역순위 평균"""
    if not results:
        return 0.0
    reciprocal_ranks = []
    for rank in results:
        if rank <= k:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def find_positive_in_documents(query: str, positive: str, documents: List[Dict], 
                               tokenizer: TokenizerWrapper, bm25: BM25Okapi, 
                               top_k: int = 50) -> int:
    """
    질문으로 검색하여 positive passage가 포함된 문서의 순위를 찾음
    Returns: 순위 (1-based), 없으면 999
    """
    # 질문 토크나이징
    query_tokens = tokenizer.tokenize(query)
    
    # BM25 검색
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # positive passage의 핵심 키워드 추출 (간단한 방법: positive를 토크나이징)
    positive_tokens = set(tokenizer.tokenize(positive))
    
    # 상위 K개 문서 중 positive 키워드가 많이 포함된 문서 찾기
    best_match_rank = None
    for rank, idx in enumerate(top_indices, start=1):
        doc_text = documents[idx]['text']
        doc_tokens = set(tokenizer.tokenize(doc_text))
        
        # 키워드 겹침 비율 계산
        overlap = len(positive_tokens & doc_tokens)
        if overlap >= len(positive_tokens) * 0.3:  # 30% 이상 겹치면 정답으로 간주
            best_match_rank = rank
            break
    
    return best_match_rank if best_match_rank else 999


def evaluate_tokenizer(tokenizer: TokenizerWrapper, documents: List[Dict], 
                       eval_dataset: List[Dict]) -> Dict:
    """특정 토크나이저로 BM25 검색 성능 평가"""
    print(f"\n{'='*60}")
    print(f"Testing Tokenizer: {tokenizer.name}")
    print(f"{'='*60}")
    
    # 1. 문서 토크나이징 및 BM25 인덱스 구축
    print("Building BM25 index...")
    start_time = time.time()
    
    tokenized_docs = []
    for doc in documents:
        tokens = tokenizer.tokenize(doc['text'])
        tokenized_docs.append(tokens)
    
    bm25 = BM25Okapi(tokenized_docs)
    index_time = time.time() - start_time
    print(f"Index built in {index_time:.2f}s")
    
    # 2. 평가 데이터셋으로 검색 테스트
    print("Running retrieval tests...")
    ranks = []
    
    for i, item in enumerate(eval_dataset):
        query = item['query']
        positive = item['positive']
        
        rank = find_positive_in_documents(query, positive, documents, tokenizer, bm25)
        ranks.append(rank)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(eval_dataset)} queries...")
    
    # 3. 메트릭 계산
    recall_at_5 = calculate_recall_at_k(ranks, k=5)
    mrr_at_5 = calculate_mrr_at_k(ranks, k=5)
    
    # 4. 결과 반환
    return {
        'tokenizer': tokenizer.name,
        'index_time': index_time,
        'recall_at_5': recall_at_5,
        'mrr_at_5': mrr_at_5,
        'ranks': ranks
    }


def main():
    # 경로 설정
    DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
    EVAL_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/evaluation_dataset.json"
    
    print("Loading documents and evaluation dataset...")
    documents = load_documents(DATA_PATH)
    eval_dataset = load_evaluation_dataset(EVAL_PATH)
    print(f"Loaded {len(documents)} documents and {len(eval_dataset)} test queries")
    
    # 토크나이저 초기화
    tokenizers = []
    
    if KIWI_AVAILABLE:
        try:
            kiwi = Kiwi()
            tokenizers.append(TokenizerWrapper("Kiwi", kiwi))
        except Exception as e:
            print(f"Failed to initialize Kiwi: {e}")
    
    if MECAB_KO_AVAILABLE:
        try:
            mecab = MeCabKo()
            tokenizers.append(TokenizerWrapper("Mecab", mecab))
        except Exception as e:
            print(f"Failed to initialize Mecab: {e}")
    
    if KONLPY_OKT_AVAILABLE:
        try:
            okt = Okt()
            tokenizers.append(TokenizerWrapper("Okt", okt))
        except Exception as e:
            print(f"Failed to initialize Okt: {e}")
    
    if not tokenizers:
        print("ERROR: No tokenizers available!")
        return
    
    # 각 토크나이저 평가
    results = []
    for tokenizer in tokenizers:
        try:
            result = evaluate_tokenizer(tokenizer, documents, eval_dataset)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {tokenizer.name}: {e}")
    
    # 결과 출력
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Tokenizer':<15} | {'Recall@5':<10} | {'MRR@5':<10} | {'Index Time':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['tokenizer']:<15} | {result['recall_at_5']:.4f}     | "
              f"{result['mrr_at_5']:.4f}     | {result['index_time']:.2f}s")
    
    # 최고 성능 토크나이저 선정 (MRR@5 우선, 동일하면 Recall@5)
    if results:
        best = max(results, key=lambda x: (x['mrr_at_5'], x['recall_at_5']))
        print(f"\n🏆 Best Tokenizer: {best['tokenizer']}")
        print(f"   - Recall@5: {best['recall_at_5']:.4f}")
        print(f"   - MRR@5: {best['mrr_at_5']:.4f}")
        print(f"   - Index Time: {best['index_time']:.2f}s")


if __name__ == "__main__":
    main()

