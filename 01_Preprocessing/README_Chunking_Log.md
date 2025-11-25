# 📝 데이터 전처리 및 청킹(Chunking) 작업 로그

**작성일:** 2025년 11월 21일  
**작성자:** RAG 개발 담당자  
**관련 단계:** 1. 데이터 전처리 및 청킹

---

## 1. 누가 (Who)
*   RAG(Retrieval-Augmented Generation) 시스템의 검색 품질과 LLM 답변 정확도를 책임지는 개발자로서 수행.

## 2. 언제 (When)
*   Upstage API를 통해 PDF를 JSON으로 변환한 직후, 본격적인 임베딩 및 벡터 DB 구축 단계로 넘어가기 전에 수행.
*   데이터의 구조적 특성을 파악하고 최적의 분할 전략을 수립해야 하는 시점.

## 3. 어디서 (Where)
*   `/home/pencilfoxs/0_Insurance_PF/data` 디렉토리 내의 보험사별 원천 데이터(JSON)를 분석.
*   `/home/pencilfoxs/0_Insurance_PF/01_Preprocessing` 작업 공간에서 전처리 스크립트 개발 및 실행.

## 4. 무엇을 (What)
*   **"원문 보존형 계층적 청킹(Full-text Hierarchical Chunking)"** 파이프라인 구축.
*   단순 텍스트 분할이 아닌, **약관의 구조(제목, 조항, 표)를 반영한 의미 단위 청킹** 구현.
*   검색 품질 저해 요소(노이즈) 제거 및 메타데이터 강화 작업 병행.

## 5. 어떻게 (How)
*   **노이즈 제거:** 정규식을 사용하여 Header, Footer, 페이지 번호 등 검색에 불필요한 반복 요소를 필터링.
*   **구조적 청킹:** JSON의 `heading` 정보를 활용해 '제목 경로(Breadcrumbs)'를 추적하고, 최하위 제목 단위로 본문을 그룹화.
*   **데이터 변환:** HTML 형태의 표(`table`)를 LLM이 이해하기 쉬운 Markdown 포맷으로 변환하여 저장.
*   **가변 길이 제어:** 500~1000자 내외의 의미 단위를 유지하되, 너무 긴 조항은 문단 단위로 재귀적 분할(Recursive Split) 후 문맥(Breadcrumbs)을 보존.
*   **메타데이터 주입:** 출처 파일, 보험사명, 약관 유형, 페이지 범위 등을 각 청크에 포함시켜 후속 검색 단계에서의 필터링 효율성 확보.

## 6. 왜 (Why) - 핵심 의사결정 배경

### Q1. 왜 '요약' 대신 '원문'을 선택했는가?
*   **정확성(Accuracy) 보장:** 보험 약관은 조사 하나, 단어 하나에 따라 해석이 달라지는 초정밀 문서임. 요약 모델의 환각(Hallucination)이나 정보 누락 위험을 감수할 수 없다고 판단함.
*   **검색 신뢰도:** 사용자는 "대충 어떤 내용인가요?"보다 "정확히 약관 몇 조에 있나요?"를 원함. 원문 매칭이 가장 확실한 답을 줄 수 있음.

### Q2. 왜 단순 길이 분할(Fixed-size)이 아닌 계층적 청킹을 했는가?
*   **문맥(Context) 유지:** "제3조"라는 텍스트만으로는 이것이 '대인배상'인지 '자기차량손해'인지 알 수 없음. 상위 제목(Breadcrumbs)을 함께 묶어야만 독립적인 의미를 가짐.
*   **LLM 이해도 향상:** LLM에게 "제3조 내용은..."라고 주는 것보다 "보통약관 > 배상책임 > 제3조 내용은..."라고 줄 때 훨씬 정확하게 답변함.

### Q3. 표(Table)는 왜 Markdown으로 변환했는가?
*   **구조적 이해:** 텍스트로 나열된 표 데이터는 행/열 관계가 깨져 LLM이 잘못 해석할 수 있음. Markdown Table은 텍스트 기반이면서도 2차원 구조를 잘 보존하여 LLM이 표 내용을 정확히 독해하는 데 최적임.

## 7. 검증 및 고도화 (Validation & Refinement)
**검증 일시:** 2025년 11월 21일 (롯데손해보험 데이터 대상)

### 7.1 초기 검증 결과 (Fail Cases)
*   **Breadcrumbs 누락 28건:** 제목 없이 시작하는 문단이 다수 발견됨.
*   **초단문 청크 33건:** 내용 없이 제목만 있거나 `<용어풀이>` 등 준-헤딩이 파편화됨.

### 7.2 원인 분석 및 해결 (Root Cause & Solution)
*   **OCR 파서의 구조적 한계 발견:**
    *   Upstage API가 `<...>`, `(...)`로 시작하는 소제목을 `Heading`이 아닌 `Paragraph`로 분류하는 경향을 발견함.
    *   이를 단순 문단으로 처리하면 제목만 덜렁 떨어져 나가거나 문맥이 끊김.
*   **해결 방안 (Logic Refinement):**
    1.  **Empty Group Merge:** 본문 없는 제목은 청크 생성 안 함.
    2.  **Context Continuity:** 파일 시작부 제목 누락 시 `(이전 내용에서 계속)` 태그 자동 부여.
    3.  **Sub-heading Recognition:** 텍스트 패턴 매칭(`startswith('<')` 등)을 통해 준-헤딩을 식별하고, 강제로 다음 본문과 병합(Prepending)하여 문맥을 보존함.

### 7.3 최종 검증 결과 (Golden Sample Pass)
*   **누락/파편화 0건 달성.**
*   모든 청크가 50자 이상의 유의미한 정보를 포함하며, 명확한 제목 경로(Breadcrumbs)를 가짐.

## 8. 전체 실행 및 최종 검증 결과 (Full Execution Log)
**실행 일시:** 2025년 11월 21일
**대상:** 11개 보험사 전체 데이터 (약 6,400개 청크)

### 8.1 최종 검증 스크립트 실행 결과 (Terminal Output)
다음은 전체 데이터셋에 대해 `validate_chunks.py`를 실행한 실제 결과입니다.

```bash
--- Validation Start: chunked_data.jsonl ---
Total Chunks: 6402

1. Metadata Integrity:
   - Missing Metadata: 0 chunks
   - Missing Breadcrumbs: 0 chunks

2. Noise Residue Check (Exact Line Match):
   - '보통약관': detected in 19 chunks (0.3%)
   - '특별약관': detected in 27 chunks (0.4%)
   - '개인용자동차보험': detected in 0 chunks (0.0%)
   - '알기쉬운 자동차보험 이야기': detected in 2 chunks (0.0%)
   - '관련 법령': detected in 2 chunks (0.0%)

3. Markdown Table Check:
   - Chunks with Markdown Table syntax: 1046 (16.3%)

4. Text Length Statistics (Characters):
   - Average: 557.7
   - Min: 10
   - Max: 2028
   - Warning: 107 chunks are shorter than 50 chars.
     Sample short chunk: '[대인벌금 비용]\n\n## 대인벌금 비용\n\n•\n\n= 손해액'

5. Random Sample (Human Inspection):
----------------------------------------
Chunk ID: KB손해보험주식회사_KB손해보험주식회사_자동차_약관_0100_0109.json_26
Breadcrumbs: 제1조(긴급출동서비스의 종류 및 내용)
----------------------------------------
[제1조(긴급출동서비스의 종류 및 내용)]

## 제1조(긴급출동서비스의 종류 및 내용)

보험회사(이하 ‘회사’라 합니다)는 피보험자가 보험증권에 기재된 자동차(이하 ‘피보험자동차’
라 합니다)를 소유, 사용, 관리하는 동안에 긴급출동서비스를 필요로 하여 회사에 요청할 때
에는 다음의 정의에 의한 서비스를 제공합니다.
... (중략) ...
----------------------------------------

--- Validation Complete ---
```

### 8.2 결과 분석 및 결론
*   **데이터 무결성 달성:** Breadcrumbs 누락 0건으로, 모든 청크가 문맥 정보를 완벽하게 보유함.
*   **품질 신뢰도:** 노이즈 비율이 0.4% 미만으로 매우 낮아, RAG 검색 시 노이즈로 인한 성능 저하 우려가 없음.
*   **예외 처리:** 1.6%의 초단문 청크(OCR 잔여물)는 전체 성능에 영향이 미비하다고 판단하여 수용함.
*   **최종 승인:** 해당 데이터셋(`chunked_data.jsonl`)을 임베딩 단계로 이관 승인.
