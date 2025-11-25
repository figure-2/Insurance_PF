# Fine-Tuning v4 및 평가 전략 (Post-Generation Plan)

**작성 일시:** 2025-11-25 03:00  
**작성자:** AI Assistant / User  
**관련 단계:** Dataset Generation v4 (Completed), Fine-Tuning v4, Evaluation  

---

## 1. 개요 (Overview)
본 문서는 v4 데이터셋(Negative, Calculation, Article 강화) 생성이 완료된 후, **데이터 품질 검증(QA)부터 파인튜닝, 평가, 그리고 RAG 파이프라인 통합까지의 전체 전략**을 기술한다. 
단순한 모델 학습 수행을 넘어, **현업 RAG/LLM 전문가 및 채용 평가관의 관점**에서 "왜(Why) 이렇게 설계했는지", "무엇이(What) 개선되었는지"를 증명하는 것을 목표로 한다.

---

## 2. [Phase 1] 데이터 품질 검증 (Data QA)
**목표:** Garbage In, Garbage Out을 방지하고 학습 데이터의 신뢰성을 확보한다.

### 2.1. 데이터 무결성 및 정제 (Integrity Check)
*   **Why (왜):** 데이터 생성 중단 및 재개(Resume) 과정에서 JSON 구조가 손상되거나 필드가 누락될 가능성이 있음.
*   **How (어떻게):**
    *   **JSON 문법 검증:** 파싱 에러가 발생하는지 전수 검사.
    *   **필드 무결성:** `instruction`, `input`, `output`, `type` 필드 존재 여부 확인.
    *   **중복 제거:** `instruction` 기준 중복 데이터(Deduplication) 제거.

### 2.2. 클래스 불균형 점검 (Class Distribution)
*   **Why (왜):** 특정 유형(Negative, Calculation)의 데이터가 부족하면 해당 능력(거절하기, 계산하기)이 학습되지 않음.
*   **How (어떻게):**
    *   생성된 데이터의 `type`별 분포(Negative, Positive, Calculation, Article, Synonym) 시각화.
    *   분포 불균형 확인 시, 학습 단계에서 **Stratified Split** 필수 적용 명시.

### 2.3. 휴먼 인 더 루프 (Human-in-the-loop)
*   **Why (왜):** 합성 데이터(Synthetic Data) 자체의 오류(Hallucination) 가능성 배제 불가.
*   **How (어떻게):**
    *   **Negative Sample:** "약관에 내용이 없습니다"라고 정확히 방어하는지 무작위 20개 검수.
    *   **Calculation Sample:** 계산 수식과 논리가 정확한지 무작위 10개 검수.

---

## 3. [Phase 2] 파인튜닝 전략 (Fine-Tuning v4)
**목표:** 특수 목적(거절, 계산, 근거 제시)에 최적화된 도메인 특화 모델 학습.

### 3.1. 데이터 분할 (Stratified Split)
*   **전략:** Train(80%) : Valid(20%)
*   **Why (근거):** 평가셋(Valid)에 모든 유형(특히 희소한 Negative/Calculation)이 골고루 포함되어야 공정한 성능 평가가 가능함.
*   **Action:** 단순 Random Split 대신, `type` 컬럼을 기준으로 한 **Stratified Shuffle Split** 적용.

### 3.2. 하이퍼파라미터 및 학습 전략
*   **Base Model:** `beomi/Llama-3-Open-Ko-8B`
*   **Method:** QLoRA (4-bit quantization)
*   **Config:**
    *   `Epoch`: 과적합 방지를 위해 Early Stopping 적용 (eval_loss 모니터링).
    *   `Objective`: **Hallucination 방지**("모르는 것은 모른다고 답하기") 및 **Evidence 제시**("약관 제O조에 따라") 능력 강화.

---

## 4. [Phase 3] 다각도 평가 및 비교 (Evaluation)
**목표:** v3(기존) vs v4(신규) 모델의 성능 차이를 정량/정성적으로 증명하여 포트폴리오 핵심 성과로 제시.

### 4.1. 정량적 평가 (Quantitative)
*   **지표 (Metrics):**
    *   **Faithfulness (충실성):** 약관(Context)에 기반한 답변인가? (Negative 데이터 평가 핵심)
    *   **Answer Relevance (관련성):** 질문 의도에 부합하는가?
    *   **Accuracy (유형별):** 특히 Negative 질문에 대한 방어 성공률(Attack Success Rate 역수).

### 4.2. 정성적 평가 (Qualitative - Case Study)
*   **비교군:** Base Model vs Fine-Tuned v3 vs **Fine-Tuned v4**
*   **Case 1 (Negative):** 약관에 없는 내용 질문 시 v3(환각) vs v4(방어) 비교 캡처.
*   **Case 2 (Calculation):** 복잡한 보험금 계산 논리의 정확성 비교.
*   **Case 3 (Evidence):** 근거 조항 명시 여부 비교.

---

## 5. [Phase 4] RAG 파이프라인 통합
**목표:** 단순 모델 성능을 넘어 실제 서비스 환경(검색 포함)에서의 동작 검증.

### 5.1. 하이브리드 검색 연동 (Hybrid Search)
*   **구성:** Dense Retriever (Embedding) + Sparse Retriever (BM25)
*   **역할:** 파인튜닝된 v4 모델이 검색된 Context를 얼마나 잘 활용(In-Context Learning)하는지 검증.

### 5.2. End-to-End 테스트
*   **시나리오:** 사용자 질문 입력 -> 검색 -> 답변 생성 -> 응답.
*   **체크리스트:** 전체 응답 속도(Latency), 검색 정확도(Top-k Recall), 최종 답변 품질.

---

## 6. [Phase 5] 문서화 및 포트폴리오 (Documentation)
**목표:** '문제 해결 역량'을 보여주는 엔지니어링 문서 작성.

1.  **실험 리포트 (`README_FineTuning_Log.md` 업데이트):**
    *   **가설:** "Negative 데이터를 학습하면 환각이 줄어들 것이다."
    *   **검증:** v3 vs v4 비교 실험 결과표 제시.
    *   **결론:** "v4 모델 채택 및 근거."
2.  **기술 명세서 (`SPEC_FineTuning_Strategy.md`):**
    *   데이터 스키마, 학습 파라미터, 평가 방법론 상세 기술.

---

## 7. Action Items (Immediate)
1.  **데이터 병합 및 검증 스크립트 작성:** `validate_and_merge_v4.py`
2.  **Stratified Split 스크립트 작성:** `split_dataset_v4.py`
3.  **v4 학습 스크립트 준비:** `train_qlora_v4.py`
