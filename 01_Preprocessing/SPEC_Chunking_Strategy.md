# 🛠️ 최종 청킹 전략 명세서 (RAG Chunking Blueprint)

이 문서는 "정확도(Accuracy)"와 "구조적 완결성(Structural Integrity)"을 최우선으로 하는 청킹 파이프라인의 구현 스펙을 정의합니다.

---

## 1. 전처리 및 정제 (Pre-processing & Cleaning)
**목표:** 검색 품질을 떨어뜨리는 노이즈를 원천 차단.

*   **헤더/푸터 제거:** 
    *   JSON `elements` 순회 시 `category`가 `header`, `footer`인 항목 Drop.
*   **페이지 번호 제거:** 
    *   `category`가 `paragraph`여도 정규식(`^\d+$`, `^- \d+ -$`) 매칭 시 제거.
*   **좌표 정렬:** 
    *   `page` 오름차순 -> `y` 좌표 오름차순 정렬로 읽는 순서(Reading Order) 보장.

## 2. 콘텐츠 변환 (Content Transformation)
**목표:** LLM 친화적 포맷으로 표준화.

*   **표 (Table):**
    *   `text` 필드 대신 **`html` 필드** 사용 필수.
    *   `markdownify` 라이브러리 등을 사용해 **Markdown Table**로 변환.
    *   표 바로 위에 직전 Heading이나 캡션을 prepend하여 문맥 보강.
*   **이미지 (Figure):**
    *   OCR 텍스트 존재 시 포함, 부재 시 `[이미지: 페이지 N 참조]` 플레이스홀더 삽입.

## 3. 계층적 맥락 추적 (Hierarchical Context Tracking)
**목표:** 모든 텍스트 조각의 소속을 명시.

*   **브레드크럼(Breadcrumbs):**
    *   문서 순회 중 제목 경로를 Stack으로 관리 (예: `['보통약관', '제2편', '제3조']`).
    *   모든 청크 생성 시 현재 시점의 Breadcrumbs를 메타데이터 및 본문 선두에 포함.
*   **약관 유형 태깅:**
    *   최상위 헤딩(H1) 기반으로 `보통약관`, `특별약관` 구분하여 변수 저장.

## 4. 청킹 및 병합 로직 (Chunking & Merging Logic) - v2.0
**목표:** 의미 단위 보존 vs 검색 효율성 균형.

*   **기본 그룹화:** 최하위 제목(Heading) 단위로 하위 본문/표를 하나의 그룹으로 묶음.
*   **빈 그룹 병합 (Empty Group Merge):**
    *   새로운 제목이 등장했을 때, 이전 그룹에 **본문 내용(Content)이 없다면** 청크를 생성하지 않고 병합하거나 제목만 교체함. (제목만 있는 텅 빈 청크 방지)
*   **준-헤딩 처리 (Sub-heading Recognition):**
    *   **조건:** `category`가 `paragraph`이지만, 텍스트가 `<...>`, `(...)`, `[...]`, `※` 등으로 시작하고 길이가 **40자 미만**인 경우.
    *   **처리:** 독립 청크로 분리하지 않고, **현재 그룹에 줄바꿈(`\n### `)과 함께 추가**하여 다음 본문과 한 덩어리로 만듦.
*   **연속성 보장 (Context Continuity):**
    *   파일 시작 시 제목 없이 본문이 나오면 `breadcrumbs = ["(이전 내용에서 계속)"]`을 강제 할당하여 Metadata 누락 방지.
*   **길이 제어 (Token Sizing):**
    *   **Target:** 300 ~ 800 Tokens (약 500~1500자)
    *   **Too Long (>1000):** 문단 단위 재귀적 분할(Recursive Split) 및 Breadcrumbs 명시.

## 5. 메타데이터 스키마 (Metadata Schema)
**목표:** 풍부한 필터링 정보 제공.

```json
{
  "source": "롯데손해보험_자동차_약관.pdf",
  "company": "롯데손해보험주식회사",
  "chunk_id": "lotte_auto_001",
  "category": "보통약관",
  "breadcrumbs": "보통약관 > 제2편 > 제3조(보상하는 손해)",
  "page_range": [38, 39],
  "token_count": 450,
  "contains_table": true,
  "is_split": false
}
```

## 6. 검증 기준 (Validation Criteria) - v1.0
**목표:** 아래 기준을 **100% 만족**해야만 임베딩 단계로 넘어갈 수 있다.

1.  **Metadata Integrity:**
    *   모든 청크는 비어있지 않은 `breadcrumbs`를 가져야 한다. (Null/Empty 불허)
2.  **Minimum Information:**
    *   모든 청크의 텍스트 길이는 최소 **50자 이상**이어야 한다. (특수문자 제외)
3.  **Noise Free:**
    *   `보통약관`, `특별약관`, `페이지 번호` 등 반복 헤더/푸터가 본문에 단독 라인으로 존재하면 안 된다. (허용률 < 1%)
4.  **Table Quality:**
    *   `metadata.contains_table=true`인 청크는 반드시 Markdown Table 문법(`|---|`)을 포함해야 한다.
