# 🛠️ 임베딩 및 벡터 DB 구축 전략 (Technical Spec)

## 1. 개요 (Overview)
이 문서는 전처리된 보험 약관 데이터를 벡터화(Embedding)하고, 이를 검색 가능한 벡터 데이터베이스(Vector DB)에 저장하는 기술적 명세와 스키마를 정의한다.

## 2. 임베딩 모델 (Embedding Model)
*   **모델명:** **`jhgan/ko-sroberta-multitask`**
*   **선정 이유:** 30개 FAQ 기반 비교 평가 결과, 한국어 약관의 정답/오답 변별력(Separation)이 가장 우수하며 처리 속도가 압도적으로 빠름.
*   **차원(Dimension):** 768 (Dense Vector)
*   **Max Token Length:** 128 (기본값) -> 긴 텍스트는 자동 Truncate 되거나 Chunking 단계에서 제어됨.
*   **라이브러리:** `langchain-huggingface` (`HuggingFaceEmbeddings`)
*   **설정:**
    *   `device`: `torch.cuda.is_available()`로 자동 감지 (GPU 우선, 없으면 CPU로 Fallback)
    *   `normalize_embeddings`: `True` (Cosine Similarity 사용을 위해 필수)
*   **구현 파일:** `create_vector_db.py`, `validate_vector_db.py`

## 3. 벡터 데이터베이스 (Vector DB)
*   **엔진:** **ChromaDB** (Local Persistence)
*   **저장 경로:** `/home/pencilfoxs/0_Insurance_PF/02_Embedding/chroma_db`
*   **컬렉션 이름:** `insurance_policies`
*   **Distance Function:** Cosine Similarity

## 4. 데이터 스키마 (Data Schema)

ChromaDB에 저장되는 `Document` 객체의 구조는 다음과 같다.

### 4.1 Document Content (`page_content`)
*   **Source:** `chunked_data.jsonl`의 `text` 필드.
*   **Format:** Breadcrumbs(제목 경로)가 포함된 Markdown 텍스트.
    ```markdown
    [보통약관 > 제2편 > 제3조]
    
    ## 제3조(보상하지 않는 손해)
    ... 본문 ...
    ```

### 4.2 Metadata (`metadata`)
검색 시 필터링(Pre-filtering) 및 답변 생성 시 출처 표기에 사용되는 메타데이터.

| 필드명 | 타입 | 설명 | 예시 |
| :--- | :--- | :--- | :--- |
| `chunk_id` | `str` | 청크 고유 식별자 | `롯데손해_..._10` |
| `company` | `str` | 보험사 이름 (필터링 핵심 키) | `롯데손해보험주식회사` |
| `policy_type` | `str` | 약관 대분류 | `보통약관`, `특별약관` |
| `breadcrumbs` | `str` | 문서 구조 경로 | `보통약관 > 배상책임 > 제1조` |
| `source` | `str` | 원본 파일 경로 | `.../data.json` |
| `page_start` | `int` | 시작 페이지 번호 | `5` |
| `page_end` | `int` | 끝 페이지 번호 | `6` |
| `token_count` | `int` | 텍스트 토큰 수 | `250` |

> **Note:** ChromaDB는 메타데이터 값으로 `List`나 `Dict` 타입을 직접 지원하지 않는 경우가 많으므로, `page_range` 리스트는 `page_start`, `page_end`로 평탄화(Flatten)하여 저장한다.

## 5. 인제스천 파이프라인 (Ingestion Pipeline)
1.  **Load:** `chunked_data.jsonl` 라인 단위 Read.
2.  **Transform:** JSON 파싱 -> Metadata Flattening -> `Document` 객체 생성.
3.  **Device Detection:** `torch.cuda.is_available()`로 GPU/CPU 자동 감지.
4.  **Embed:** `jhgan/ko-sroberta-multitask`를 통해 텍스트를 768차원 벡터로 변환.
5.  **Upsert:** ChromaDB에 배치 단위(Batch Size: 100)로 저장.
6.  **Persist:** 로컬 디스크(`sqlite3`)에 영구 저장.

### 5.1 실제 구축 결과 (Implementation Results)
*   **총 문서 수:** 6,402개
*   **소요 시간:** 47.93초 (GPU A100-SXM4-40GB)
*   **평균 처리 속도:** 0.007초/doc
*   **저장 위치:** `/home/pencilfoxs/0_Insurance_PF/02_Embedding/chroma_db`

## 6. 검색 전략 (Retrieval Strategy)
*   **기본 검색:** Query Embedding과 Doc Embedding 간의 Cosine Similarity Top-K.
*   **메타데이터 필터링:** 사용자가 특정 보험사를 선택할 경우 `filter={"company": "..."}` 적용.
*   **앙상블(Ensemble) 고려:** `ko-sroberta`가 의미 파악엔 강하나 키워드 매칭이 약할 수 있으므로, 필요 시 BM25(키워드 검색)와 결합하는 Hybrid Search를 추후 고려함.
