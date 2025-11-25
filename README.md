# 🚗 개인용 자동차 보험 특화 LLM (Auto Insurance Specialist Agent)

> **"사용자의 모호한 보험 질문을 정확한 약관 정보로 답변하는 RAG 기반 전문가 에이전트"**

---

## 📋 프로젝트 개요

복잡하고 방대한 자동차 보험 약관을 일반 사용자가 이해하기 쉽게 설명해주는 LLM 서비스입니다. 단순히 문서를 검색하는 것을 넘어, **질문의 의도를 파악하고 정확한 근거(약관 조항)를 기반으로 답변**하도록 설계되었습니다.

### 핵심 목표
- **정확성**: 보험 약관의 정확한 조항을 기반으로 답변 제공
- **이해도**: 전문 용어를 일반인이 이해하기 쉬운 언어로 변환
- **신뢰성**: Hallucination 방지 및 출처 명시를 통한 투명한 답변

### 주요 기능
- 보험 약관 기반 질의응답 (Q&A)
- 특약 추천 및 보장 범위 분석
- 약관 조항 검색 및 해석

---

## 🔬 문제 해결 과정 (Engineering Log)

본 프로젝트는 **가설(Hypothesis) → 실험(Experiment) → 검증(Validation) → 의사결정(Conclusion)**의 체계적인 프로세스를 준수하며 진행되었습니다. 각 단계의 상세한 엔지니어링 로그는 아래 링크에서 확인할 수 있습니다.

| 단계 | 주요 내용 | 상세 로그 (Click) | 기술 명세서 |
|:---:|:---|:---|:---|
| **Step 1** | **데이터 전처리**<br>약관 데이터의 구조적 특성을 고려한 Chunking 전략 수립 | [📄 01_Preprocessing Log](./01_Preprocessing/README_Chunking_Log.md) | [📋 SPEC](./01_Preprocessing/SPEC_Chunking_Strategy.md) |
| **Step 2** | **임베딩 모델 선정**<br>한국어 보험 용어에 특화된 임베딩 모델 비교 실험 | [📄 02_Embedding Log](./02_Embedding/README_Embedding_Log.md) | [📋 SPEC](./02_Embedding/SPEC_Embedding_Strategy.md) |
| **Step 3** | **RAG 파이프라인**<br>Retrieval 정확도 향상을 위한 검색 기법 고도화 | [📄 03_Retrieval Log](./03_Retrieval/README_Retrieval_Log.md) | [📋 SPEC](./03_Retrieval/SPEC_Retrieval_Strategy.md) |
| **Step 4** | **LLM & 프롬프트**<br>Hallucination 방지를 위한 프롬프트 엔지니어링 | [📄 04_LLM Log](./04_LLM/README_LLM_Log.md) | - |
| **Step 5** | **Fine-Tuning**<br>도메인 특화 성능 향상을 위한 학습 실험 | [📄 05_FineTuning Log](./05_FineTuning/README_FineTuning_Log.md) | [📋 SPEC](./05_FineTuning/SPEC_FineTuning_Strategy.md) |
| **Step 6** | **성능 평가**<br>정량적(Metrics) 및 정성적(Human Eval) 평가 결과 | [📄 06_Evaluation Log](./06_Evaluation/README_Evaluation_Log.md) | - |

> *각 로그 파일은 **육하원칙(Who, When, Where, What, How, Why)**에 기반하여 작성되었으며, 실험 설계, 검증 결과, 의사결정 과정을 상세히 기록하고 있습니다.*

---

## 🎯 핵심 성과 및 의사결정 (Key Metrics)

### 1. 데이터 전처리 (Chunking)
- **전략**: 원문 보존형 계층적 청킹 (Full-text Hierarchical Chunking)
- **결과**: 약관의 구조(제목, 조항, 표)를 반영한 의미 단위 청킹 구현
- **개선**: 검색 품질 저해 요소(노이즈) 제거 및 메타데이터 강화

### 2. 임베딩 모델 선정
- **후보군**: `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `jhgan/ko-sroberta-multitask`
- **최종 선택**: `jhgan/ko-sroberta-multitask` (한국어 특화 성능 우수)
- **근거**: 정량 평가(Recall, MRR) 및 정성 평가(20개 이상 샘플 검증) 결과

### 3. Retrieval 전략
- **하이브리드 검색**: Dense Retrieval + BM25 (Sparse Retrieval)
- **성능 개선**: 단일 검색 방식 대비 정확도 향상
- **최종 결정**: Sparse Only 전략 채택 (도메인 특성에 최적화)

### 4. LLM & 프롬프트 엔지니어링
- **Base Model**: `beomi/Llama-3-Open-Ko-8B`
- **프롬프트 전략**: CoT (Chain of Thought) 적용
- **Hallucination 감소**: 프롬프트 최적화를 통한 오답률 감소

### 5. Fine-Tuning
- **방법**: QLoRA (Quantized LoRA)
- **데이터**: 도메인 특화 Instruction 데이터 생성
- **성능**: Base Model 대비 도메인 특화 성능 향상

### 6. 최종 평가
- **평가 지표**: 정량 평가 (Recall, MRR, Accuracy) + 정성 평가 (Human Eval)
- **실패 분석**: 엣지 케이스 분석 및 개선 방향 도출

---

## 📁 디렉토리 구조 (Structure)

```
.
├── 00_chat/                    # 챗봇 UI 및 인터페이스 코드
├── 01_Preprocessing/           # 데이터 정제 및 청킹 (Chunking) 전략
│   ├── README_Chunking_Log.md  # 작업 로그
│   ├── SPEC_Chunking_Strategy.md # 기술 명세서
│   └── ...
├── 02_Embedding/               # Vector DB 구축 및 임베딩 실험
│   ├── README_Embedding_Log.md
│   ├── SPEC_Embedding_Strategy.md
│   └── ...
├── 02_Embedding_VectorDB/      # 벡터 DB 관련 추가 작업
├── 03_RAG_Pipeline/            # RAG 파이프라인 통합
├── 03_Retrieval/               # 검색(Retrieval) 및 답변 생성 로직
│   ├── README_Retrieval_Log.md
│   ├── SPEC_Retrieval_Strategy.md
│   └── ...
├── 04_LLM/                     # 프롬프트 엔지니어링 및 모델 관리
│   ├── README_LLM_Log.md
│   └── ...
├── 04_Synthetic_Data/          # 합성 데이터 생성
├── 05_FineTuning/              # 모델 미세 조정 (LoRA 등)
│   ├── README_FineTuning_Log.md
│   ├── SPEC_FineTuning_Strategy.md
│   └── ...
├── 06_Evaluation/              # 성능 평가 지표 및 테스트셋
│   ├── README_Evaluation_Log.md
│   └── ...
├── data/                       # (Sample) 실험에 사용된 데이터 샘플
├── financial-agent/             # 관련 프로젝트 (참고용)
└── rule.md                     # 프로젝트 진행 원칙 (Ground Rules)
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **LLM Framework**: LangChain, LlamaIndex
- **Embedding Models**: 
  - `jhgan/ko-sroberta-multitask` (최종 선택)
  - `BAAI/bge-m3` (비교 실험)
  - `intfloat/multilingual-e5-large` (비교 실험)
- **Vector DB**: ChromaDB
- **Retrieval**: Dense Retrieval, BM25 (Sparse Retrieval)
- **LLM**: 
  - Base: `beomi/Llama-3-Open-Ko-8B`
  - Fine-Tuned: QLoRA 적용
- **Fine-Tuning**: QLoRA (Quantized LoRA)
- **UI**: Streamlit (예정)

---

## 📚 프로젝트 진행 원칙

본 프로젝트는 다음 원칙에 따라 진행되었습니다:

1. **문서화 표준 (Engineering Log)**: 각 단계 완료 시, 단순 결과가 아닌 '문제 해결 과정'을 중심으로 문서 작성
2. **육하원칙 준수**: 모든 문서는 '누가, 언제, 어디서, 무엇을, 어떻게, 왜'에 기반하여 작성
3. **실험 및 검증 원칙**: 정량 평가(Quantitative)와 정성 평가(Qualitative) 병행
4. **비교 실험 공통 규범**: 후보 선정 사유, 실험 규모, 평가 결과, 실패 분석, 최종 결론 명시

자세한 내용은 [rule.md](./rule.md)를 참고하세요.

---

## 🚀 시작하기

### 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (Fine-Tuning 및 추론 시 권장)

### 설치
```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 실행
```bash
# 각 단계별 스크립트는 해당 디렉토리의 README를 참고하세요
# 예: 01_Preprocessing/README_Chunking_Log.md
```

---

## 📝 주요 의사결정 요약

| 단계 | 의사결정 | 근거 |
|:---:|:---|:---|
| **Chunking** | 계층적 청킹 전략 채택 | 약관의 구조적 특성 반영 필요 |
| **Embedding** | `ko-sroberta-multitask` 선택 | 한국어 특화 성능 우수 (정량/정성 평가) |
| **Retrieval** | Sparse Only 전략 채택 | 도메인 특성에 최적화 (보험 용어 매칭) |
| **LLM** | `Llama-3-Open-Ko-8B` + QLoRA | 한국어 성능 및 도메인 특화 균형 |
| **Fine-Tuning** | Instruction Tuning 적용 | Base Model의 한계 보완 |

---

## 📊 평가 결과 요약

*각 단계별 상세한 평가 결과는 해당 단계의 README 로그 파일을 참고하세요.*

- **Retrieval 정확도**: [각 단계별 로그 참고]
- **Hallucination 감소**: [각 단계별 로그 참고]
- **사용자 만족도**: [각 단계별 로그 참고]

---

## 🤝 기여 및 문의

본 프로젝트는 포트폴리오 목적으로 작성되었습니다. 문의사항이 있으시면 이슈를 등록해주세요.

---

## 📄 라이선스

본 프로젝트는 개인 포트폴리오 목적으로 작성되었습니다.

---

**작성일**: 2025년 11월  
**최종 업데이트**: 2025년 11월

