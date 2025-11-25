# 📝 임베딩 모델 선정 및 벡터 DB 설계 작업 로그

**작성 일시:** 2025-11-21 16:00 (KST)  
**최종 업데이트:** 2025-11-22 00:20 (KST)  
**작성자:** RAG 개발 담당자  
**관련 단계:** 2. 임베딩 및 벡터 DB 구축 (Embedding & Vector DB)

---

## 1. 가설 (Hypothesis) & 초기 가정
*   **초기 가정:** "최신 SOTA(State-of-the-art) 모델인 `BAAI/bge-m3`가 다국어 성능이 뛰어나고 입력 길이가 길어(8192토큰), 한국어 보험 약관 RAG에도 가장 적합할 것이다."
*   **예상:** `bge-m3` > `multilingual-e5` > `ko-sroberta` 순으로 성능이 좋을 것으로 예상함.

## 2. 실험 설계 (Experiment Design)

### 2.1 비교 모델 후보군 (Candidates)
1.  **`BAAI/bge-m3`:** 다국어 모델 1위, 긴 문맥 처리 강점 (초기 유력 후보).
2.  **`intfloat/multilingual-e5-large`:** 검색(Retrieval) 특화 모델로 널리 알려짐.
3.  **`jhgan/ko-sroberta-multitask`:** 한국어 NLI/STS 특화 베스트셀러 모델 (Baseline).

### 2.2 테스트 데이터셋 (Test Dataset)
*   **1차 시도 (단일 질문):** "음주 운전하면 보험 처리 되나요?" 1개 질문으로 약식 테스트 진행. -> **통계적 신뢰성 부족으로 폐기.**
*   **2차 시도 (Golden Dataset):** 통계적 유의성 확보를 위해 **30개의 실전 FAQ 데이터셋** 구축.
    *   **구성:** 보상(8), 면책(7), 특약(5), 절차(5), 정의(5) 등 5개 카테고리.
    *   **특징:** 각 질문마다 **Hard Negative(함정 오답)**를 포함하여, 단순 키워드 매칭이 아닌 '의미적 변별력'을 집중 평가함.

## 3. 검증 결과 (Validation)

### 3.1 실험 결과 수치 (Numerical Metrics)
터미널에서 수행한 30개 FAQ 테스트 결과 요약:

| 순위 | 모델명 | Avg Pos Score (정답 유사도) | **Avg Separation (변별력)** | Time/Item (속도) |
| :--- | :--- | :--- | :--- | :--- |
| **1위** | **ko-sroberta-multitask** | 0.4481 | **0.0721** | **0.030s** |
| 2위 | multilingual-e5-large | 0.8372 | 0.0055 | 0.173s |
| 3위 | BAAI/bge-m3 | 0.5742 | **-0.0096** | 0.094s |

### 3.2 결과 분석 (Analysis)
1.  **`bge-m3`의 몰락 (Hypothesis Rejection):**
    *   초기 가설과 달리 `Separation` 점수가 음수(`-0.0096`)를 기록.
    *   이는 모델이 **정답보다 오답(함정)을 더 유사하다고 판단**하는 역전 현상이 발생했음을 의미. 한국어 약관의 미세한 부정어(~하지 않는다) 처리에 약점을 보임.
2.  **`multilingual-e5`의 한계:**
    *   모든 문서에 0.8 이상의 높은 점수를 부여하여 정답/오답 구분이 거의 불가능함(변별력 0.0055).
3.  **`ko-sroberta`의 재발견:**
    *   유일하게 양수(+)의 변별력을 보여주었으며, 속도 또한 타 모델 대비 압도적으로 빠름(0.03s).

## 4. 의사결정 (Conclusion & Pivot)

### 4.1 의사결정 변경 (Strategic Pivot)
    *   **Before:** `bge-m3` 도입 예정.
    *   **After:** **`jhgan/ko-sroberta-multitask`** 최종 선정.

### 4.2 최종 결정 논리
1.  **데이터 기반 의사결정:** "최신 모델이 좋다"는 편견을 버리고, 30개 실전 데이터 테스트 결과를 신뢰하기로 함.
2.  **변별력 우선:** RAG 시스템에서 가장 치명적인 '오답 환각(Hallucination)'을 줄이기 위해, 정답과 오답을 가장 잘 구분하는 모델을 선택함.
3.  **효율성:** 가장 빠른 속도를 제공하여 실시간 검색 서비스 구축에 유리함.

## 5. 벡터 DB 선정 (Vector DB Selection)

### 5.1 후보군 비교 (Candidates)
1.  **ChromaDB:** 로컬 파일 기반(SQLite), 설정 용이, 메타데이터 필터링 강력.
2.  **FAISS (Meta):** 대규모 데이터 검색 속도 최상, 그러나 메타데이터 관리 기능이 부족함.
3.  **Pinecone:** 완전 관리형 클라우드 서비스, 편의성은 높으나 외부 네트워크 의존 및 비용 발생 가능성.

### 5.2 선정 결과: ChromaDB
*   **이유 1 (Local-First):** 별도의 서버 구축 없이 Python 라이브러리 형태로 즉시 구동 가능하여 포트폴리오 환경에 최적.
*   **이유 2 (Metadata Filtering):** 보험사별(`company`), 약관유형별(`policy_type`) 정교한 필터링이 필수적이므로, 이를 Native로 지원하는 ChromaDB가 FAISS보다 유리함.
*   **이유 3 (LangChain Integration):** 향후 RAG 파이프라인 구축 시 LangChain과의 연동성이 가장 뛰어남.

## 6. 최종 구축 및 검증 (Final Implementation & Verification)

### 6.1 구현 이슈 및 해결 (Implementation Issues)
*   **이슈:** 클라우드(GCP) 환경 특성상 GPU 드라이버 상태에 따라 `device='cuda'` 강제 할당 시 런타임 에러 발생 가능성 존재.
*   **해결(Solution):** `torch.cuda.is_available()` 함수를 활용하여 **GPU 우선, 실패 시 CPU로 자동 전환(Fallback)**되는 하이브리드 로직 적용.
    *   수정 파일: `create_vector_db.py`, `validate_vector_db.py`
    *   코드 변경: `model_kwargs={'device': 'cuda'}` → `model_kwargs={'device': device}` (device 변수는 자동 감지)

### 6.2 최종 검증 결과 (Validation Metrics)
터미널 실행(`create_vector_db.py` 및 `validate_vector_db.py`)을 통해 얻은 확정 수치입니다.

#### 6.2.1 데이터 무결성 (Data Integrity)
*   **입력 청크 수:** 6,402개 (`chunked_data.jsonl`)
*   **적재된 문서 수:** **6,402개** (100% 일치, 누락 없음)
*   **검증 방법:** `vector_store._collection.count()` 메서드로 확인

#### 6.2.2 성능 지표 (Performance Metrics)
*   **총 소요 시간:** **47.93초** (GPU A100-SXM4-40GB 활용)
*   **평균 처리 속도:** 약 **0.007초/doc**
*   **배치 크기:** 100개 단위로 처리
*   **장치 상태:** CUDA 자동 감지 성공 (`🚀 Device Status: Using CUDA`)

#### 6.2.3 기능 검증 (Retrieval Test)
**테스트 쿼리:** "음주 운전하면 면책인가요?"

**검색 결과 (Top-3):**
1.  **Score: 0.7034** | Company: 삼성화재해상보험주식회사 | Breadcrumbs: 음주운전
    *   내용: `[음주운전]` 정의 및 면책 조항 문서
2.  **Score: 0.7580** | Company: 하나손해보험주식회사 | Breadcrumbs: 【용어풀이】
    *   내용: 무면허운전 및 음주운전 관련 용어 설명
3.  **Score: 0.8363** | Company: 삼성화재해상보험주식회사
    *   내용: "무면허운전이나 음주운전 사고 시 제한적으로 보상이 가능하며, 본인이 부담해야..."

**필터링 테스트:**
*   **조건:** `company="롯데손해보험주식회사"`
*   **결과:** 롯데손해보험 문서만 정확히 반환됨 (Score: 0.9143, 0.9950, 1.0350)

### 6.3 결론 (Conclusion)
*   **Phase 2 (Embedding & Vector DB) 구축 작업 성공적으로 완료.**
*   모든 6,402개 문서가 벡터로 변환되어 ChromaDB에 저장되었으며, 검색 및 필터링 기능이 정상 동작함을 확인.
*   다음 단계: **Phase 3 (RAG 파이프라인 구축 & LLM 연동)** 준비 완료.

---
**Next Step:** 선정된 `ko-sroberta-multitask` 모델과 `ChromaDB`를 사용하여 전체 데이터(6,402개 청크)에 대한 벡터 DB 구축이 **완료되었음.** 이제 RAG의 최종 단계인 LLM(Generator) 연동을 진행할 수 있음.
