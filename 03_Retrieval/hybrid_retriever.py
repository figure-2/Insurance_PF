"""
하이브리드 리트리버 구현
Dense (ChromaDB) + Sparse (BM25) 결합
RRF (Reciprocal Rank Fusion) 알고리즘 직접 구현
"""

import torch
from typing import List, Dict
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from bm25_retriever import MecabBM25Retriever


def get_dense_retriever(
    db_path: str = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/chroma_db",
    model_name: str = "jhgan/ko-sroberta-multitask",
    k: int = 50
):
    """
    Dense 리트리버 (ChromaDB) 생성
    
    Args:
        db_path: ChromaDB 저장 경로
        model_name: 임베딩 모델명
        k: 반환할 문서 개수
    
    Returns:
        Chroma 리트리버
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Dense Retriever: Using {device.upper()}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = Chroma(
        collection_name="insurance_policies",
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_sparse_retriever(
    index_path: str = "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl",
    k: int = 50
):
    """
    Sparse 리트리버 (BM25) 생성
    
    Args:
        index_path: BM25 인덱스 피클 파일 경로
        k: 반환할 문서 개수
    
    Returns:
        MecabBM25Retriever 인스턴스
    """
    print(f"Loading BM25 index from {index_path}...")
    retriever = MecabBM25Retriever.load_index(index_path)
    retriever.k = k  # k 값 업데이트
    return retriever


def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    k: int = 60
) -> List[Document]:
    """
    RRF (Reciprocal Rank Fusion) 알고리즘으로 여러 검색 결과를 결합
    
    Args:
        results_list: 각 리트리버의 검색 결과 리스트
        k: RRF 상수 (일반적으로 60 사용)
    
    Returns:
        결합된 문서 리스트 (중복 제거, 점수 순 정렬)
    """
    doc_scores = defaultdict(float)
    doc_map = {}  # chunk_id -> Document 매핑
    
    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            chunk_id = doc.metadata.get('chunk_id', str(id(doc)))
            score = 1.0 / (k + rank)  # RRF 점수 계산
            
            doc_scores[chunk_id] += score
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
    
    # 점수 순으로 정렬
    sorted_docs = sorted(doc_map.items(), key=lambda x: doc_scores[x[0]], reverse=True)
    
    return [doc for _, doc in sorted_docs]


class HybridRetriever(BaseRetriever):
    """하이브리드 리트리버 (Dense + Sparse)"""
    
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    k: int = 5
    rrf_k: int = 60
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """쿼리에 대한 하이브리드 검색 수행"""
        # 각 리트리버로 검색 (Top-50)
        dense_results = self.dense_retriever.invoke(query)
        sparse_results = self.sparse_retriever.invoke(query)
        
        # RRF로 결합
        fused_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k
        )
        
        # 최종 Top-K 반환
        return fused_results[:self.k]


def get_hybrid_retriever(
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    k: int = 5
):
    """
    하이브리드 리트리버 생성 (Dense + Sparse 결합)
    
    Args:
        dense_weight: Dense 리트리버 가중치 (현재는 RRF 사용으로 무시됨, 향후 확장용)
        sparse_weight: Sparse 리트리버 가중치 (현재는 RRF 사용으로 무시됨, 향후 확장용)
        k: 최종 반환할 문서 개수
    
    Returns:
        HybridRetriever 인스턴스
    """
    print("="*60)
    print("Building Hybrid Retriever (Dense + Sparse)")
    print("="*60)
    
    # Dense 리트리버 생성
    dense_retriever = get_dense_retriever(k=50)  # Top-50 추출
    
    # Sparse 리트리버 생성
    sparse_retriever = get_sparse_retriever(k=50)  # Top-50 추출
    
    # 하이브리드 리트리버 생성
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        k=k
    )
    
    print(f"Hybrid Retriever created (RRF algorithm)")
    print(f"Final top-{k} documents will be returned after RRF fusion.")
    
    return hybrid_retriever


if __name__ == "__main__":
    # 테스트 실행
    print("Testing Hybrid Retriever...")
    
    hybrid_retriever = get_hybrid_retriever(
        dense_weight=0.5,
        sparse_weight=0.5,
        k=5
    )
    
    query = "음주 운전하면 면책인가요?"
    print(f"\nTest Query: {query}")
    print("-" * 60)
    
    results = hybrid_retriever.invoke(query)
    
    print(f"\nRetrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}]")
        print(f"Company: {doc.metadata.get('company')}")
        print(f"Breadcrumbs: {doc.metadata.get('breadcrumbs')}")
        print(f"Text: {doc.page_content[:150]}...")

