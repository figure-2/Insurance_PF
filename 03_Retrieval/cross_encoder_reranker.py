"""
Cross-Encoder 기반 리랭커 모듈
 - Sparse(BM25) 검색 결과를 의미적 유사도 기준으로 재정렬
"""

import torch
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Cross-Encoder를 활용하여 Document 리스트를 재정렬하는 리랭커."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> Tuple[List[Document], List[Tuple[Document, float]]]:
        """
        주어진 문서들을 Cross-Encoder 점수 기준으로 재정렬한다.

        Returns:
            top_docs: 상위 top_k 문서 (Document 리스트)
            scored_docs: (Document, score) 튜플 리스트 (점수 내림차순)
        """
        if not documents:
            return [], []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
        return reranked_docs, scored_docs

