"""
Mecab 기반 BM25 리트리버 구현
LangChain의 BaseRetriever 인터페이스를 따르는 커스텀 리트리버
"""

import json
import pickle
import os
from typing import List, Optional, Any
from mecab import MeCab as MeCabKo
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, PrivateAttr


class MecabBM25Retriever(BaseRetriever):
    """Mecab을 사용한 BM25 리트리버"""
    
    documents: List[Document] = Field(default_factory=list)
    k: int = Field(default=5, description="Number of documents to return")
    _tokenizer: Any = PrivateAttr()
    _bm25: Any = PrivateAttr()
    _tokenized_docs: Any = PrivateAttr()
    
    def __init__(
        self,
        documents: List[Document],
        tokenizer: Optional[MeCabKo] = None,
        k: int = 5,
        bm25_index: Optional[BM25Okapi] = None,
        tokenized_docs: Optional[List[List[str]]] = None,
        **kwargs
    ):
        """
        Args:
            documents: 검색 대상 문서 리스트
            tokenizer: Mecab 토크나이저 (None이면 자동 생성)
            k: 반환할 문서 개수
            bm25_index: 미리 구축된 BM25 인덱스 (재사용 시)
            tokenized_docs: 미리 토크나이징된 문서 리스트 (재사용 시)
        """
        super().__init__(documents=documents, k=k, **kwargs)
        
        if tokenizer is None:
            self._tokenizer = MeCabKo()
        else:
            self._tokenizer = tokenizer
        
        # 인덱스가 이미 있으면 재사용, 없으면 새로 구축
        if bm25_index is not None and tokenized_docs is not None:
            self._bm25 = bm25_index
            self._tokenized_docs = tokenized_docs
        else:
            print("Building BM25 index...")
            self._tokenized_docs = [self._tokenizer.morphs(doc.page_content) for doc in documents]
            self._bm25 = BM25Okapi(self._tokenized_docs)
            print(f"BM25 index built for {len(documents)} documents.")
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """쿼리에 대한 관련 문서 검색"""
        # 쿼리 토크나이징
        query_tokens = self._tokenizer.morphs(query)
        
        # BM25 점수 계산
        scores = self._bm25.get_scores(query_tokens)
        
        # 상위 K개 문서 인덱스 추출
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
        
        # Document 객체 반환
        return [self.documents[i] for i in top_indices]
    
    def save_index(self, filepath: str):
        """BM25 인덱스와 토크나이징된 문서를 피클 파일로 저장"""
        data = {
            'bm25': self._bm25,
            'tokenized_docs': self._tokenized_docs,
            'documents': self.documents,
            'k': self.k
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"BM25 index saved to {filepath}")
    
    @classmethod
    def load_index(cls, filepath: str, tokenizer: Optional[MeCabKo] = None):
        """저장된 BM25 인덱스 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if tokenizer is None:
            tokenizer = MeCabKo()
        
        return cls(
            documents=data['documents'],
            tokenizer=tokenizer,
            k=data['k'],
            bm25_index=data['bm25'],
            tokenized_docs=data['tokenized_docs']
        )


def build_bm25_retriever_from_jsonl(
    jsonl_path: str,
    index_save_path: Optional[str] = None
) -> MecabBM25Retriever:
    """
    chunked_data.jsonl 파일에서 문서를 로드하여 BM25 리트리버 구축
    
    Args:
        jsonl_path: chunked_data.jsonl 파일 경로
        index_save_path: 인덱스를 저장할 경로 (None이면 저장 안 함)
    
    Returns:
        MecabBM25Retriever 인스턴스
    """
    documents = []
    
    print(f"Loading documents from {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            meta = item['metadata']
            chunk_id = item['chunk_id']
            
            # 메타데이터 정리
            clean_meta = {
                "source": meta.get("source", ""),
                "company": meta.get("company", ""),
                "breadcrumbs": meta.get("breadcrumbs", ""),
                "token_count": meta.get("token_count", 0),
                "chunk_id": chunk_id,
                "policy_type": meta.get("policy_type", "unknown")
            }
            
            if "page_range" in meta and isinstance(meta["page_range"], list) and len(meta["page_range"]) == 2:
                clean_meta["page_start"] = meta["page_range"][0]
                clean_meta["page_end"] = meta["page_range"][1]
            
            doc = Document(page_content=text, metadata=clean_meta)
            documents.append(doc)
    
    print(f"Loaded {len(documents)} documents.")
    
    # BM25 리트리버 생성
    retriever = MecabBM25Retriever(documents=documents, k=50)  # Top-50으로 설정 (하이브리드 결합 시 사용)
    
    # 인덱스 저장
    if index_save_path:
        retriever.save_index(index_save_path)
    
    return retriever


if __name__ == "__main__":
    # 테스트 실행
    DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
    INDEX_PATH = "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    
    # 인덱스가 이미 있으면 로드, 없으면 새로 구축
    if os.path.exists(INDEX_PATH):
        print(f"Loading existing BM25 index from {INDEX_PATH}...")
        retriever = MecabBM25Retriever.load_index(INDEX_PATH)
    else:
        print("Building new BM25 index...")
        retriever = build_bm25_retriever_from_jsonl(DATA_PATH, INDEX_PATH)
    
    # 테스트 검색
    query = "음주 운전하면 면책인가요?"
    print(f"\nTest Query: {query}")
    results = retriever.invoke(query)  # BaseRetriever의 invoke 메서드 사용
    
    print(f"\nRetrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}]")
        print(f"Company: {doc.metadata.get('company')}")
        print(f"Text: {doc.page_content[:100]}...")

