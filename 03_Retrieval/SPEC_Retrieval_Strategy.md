# 🛠️ 하이브리드 리트리버 구축 전략 (Technical Spec)

**작성 일시:** 2025-11-22 01:00 (KST)  
**최종 업데이트:** 2025-11-22 03:00 (KST)

## 1. 개요 (Overview)
이 문서는 보험 약관 RAG 시스템의 검색 성능 향상을 위해 **Dense Retrieval(의미 기반)**과 **Sparse Retrieval(키워드 기반)**을 결합한 **하이브리드 리트리버(Hybrid Retriever)**를 구축하는 기술적 명세를 정의한다.

## 2. 아키텍처 (Architecture)

### 2.1 하이브리드 리트리버 구조
```
사용자 질문 (Query)
    ↓
    ├─→ [Dense Retriever] ──┐
    │   (ko-sroberta + ChromaDB) │
    │                            │
    └─→ [Sparse Retriever] ──┐   │
        (BM25 + Tokenizer)    │   │
                              ↓   ↓
                    [Ensemble Retriever]
                    (RRF: Reciprocal Rank Fusion)
                              ↓
                    [Top-K 최종 결과]
```

### 2.2 구성 요소 (Components)

#### A. Dense Retriever (기존 구현)
*   **임베딩 모델:** `jhgan/ko-sroberta-multitask` (768차원)
*   **벡터 DB:** ChromaDB (`02_Embedding/chroma_db`)
*   **검색 방식:** Cosine Similarity

#### B. Sparse Retriever (신규 구현)
*   **알고리즘:** **BM25 (Best Matching 25)**
*   **형태소 분석기:** **Mecab (python-mecab-ko)**
    *   **선정 근거:** 정량 평가(MRR@5 0.9833, Index Time 19.29s) 및 정성 평가(20개 샘플 분석) 결과 최고 성능
    *   **특징:** 복합명사 처리 일관성 우수, 전문 용어 처리 안정적
*   **인덱스:** 메모리 기반 (`rank_bm25` 라이브러리 사용)

#### C. Ensemble Retriever (결합)
*   **방식:** **RRF (Reciprocal Rank Fusion)**
*   **가중치:** 초기에는 Dense(0.5) + Sparse(0.5) 균등 가중치로 시작, 실험 결과에 따라 조정 가능.

## 3. 데이터 준비 (Data Preparation)

### 3.1 소스 데이터
*   **입력:** `01_Preprocessing/chunked_data.jsonl` (6,402개 청크)
*   **형태:** 각 청크의 `text` 필드를 형태소 분석하여 BM25 인덱스 구축.

### 3.2 토크나이징 전략
*   **형태소 분석기:** Mecab (python-mecab-ko)
*   **메서드:** `morphs()` 메서드 사용 (형태소 단위로 분해)
*   **전처리:** 
    *   소문자 변환 없음 (한국어는 대소문자 구분 없음)
    *   특수문자 제거 없음 (약관의 조항 번호 등이 중요하므로 보존)
*   **특이사항:**
    *   숫자 분해 이슈 ("0.03" → "0 | . | 03")가 있으나, 실제 검색 성능에는 영향 미미 (정량 평가에서 증명)
    *   구어체 처리 약점이 있으나, 보험 약관은 문어체 중심이므로 문제 없음

## 4. 구현 스펙 (Implementation Spec)

### 4.1 라이브러리
*   **BM25:** `rank_bm25` (Python)
*   **형태소 분석:** `python-mecab-ko` (실험 결과 최적 성능)
*   **하이브리드 결합:** `langchain.retrievers.EnsembleRetriever`

### 4.2 성능 최적화
*   **인덱스 구축:** BM25 인덱스는 한 번만 구축하고 메모리에 유지 (재사용).
*   **병렬 처리:** Dense와 Sparse 검색을 병렬로 실행하여 속도 향상.

## 5. 검색 전략 (Retrieval Strategy)

### 5.1 기본 흐름
1.  사용자 질문 입력.
2.  **Dense 검색:** 질문을 임베딩하여 ChromaDB에서 Top-K (예: 50개) 추출.
3.  **Sparse 검색:** 질문을 토크나이징하여 BM25로 Top-K (예: 50개) 추출.
4.  **RRF 결합:** 두 결과를 RRF 알고리즘으로 합쳐서 최종 Top-K (예: 5개) 선정.

### 5.2 메타데이터 필터링
*   Dense 검색 시 ChromaDB의 메타데이터 필터(`company`, `policy_type` 등)는 그대로 활용.
*   Sparse 검색은 별도 필터링 로직 필요 (구현 시 추가).

## 6. 평가 및 검증 (Evaluation)

### 6.1 테스트 데이터셋
*   `02_Embedding/evaluation_dataset.json` (30개 FAQ)

### 6.2 평가 지표
*   **Recall@5:** 정답 포함 여부
*   **MRR@5:** 정답 순위
*   **Execution Time:** 검색 속도

---

## 7. 최종 구현 결과 (Final Implementation Results)

### 7.1 성능 비교 결과
*   **Sparse Only (Mecab + BM25):** Recall@5 1.0000, MRR@5 0.9833 (최고 성능)
*   **Hybrid (Dense + Sparse):** Recall@5 1.0000, MRR@5 0.9444
*   **Dense Only:** Recall@5 0.7000, MRR@5 0.5483

### 7.2 최종 선정
*   **프로덕션 리트리버:** **Sparse Only (Mecab + BM25)**
*   **구현 파일:** `bm25_retriever.py`
*   **인덱스 파일:** `bm25_index.pkl` (재사용 가능)

---

**Next Step:** Phase 3 (LLM 연동) 준비 완료. Sparse 리트리버를 사용하여 RAG 파이프라인 구축.

