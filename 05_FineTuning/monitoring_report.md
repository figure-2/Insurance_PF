# 백그라운드 작업 모니터링 리포트

**생성 일시:** 2025-11-24 15:41 (KST)  
**리포트 버전:** 1.0

---

## 📊 전체 작업 현황 요약

| 항목 | 작업 1: v4 데이터셋 생성 | 작업 2: Ragas 평가 |
|:---|:---|:---|
| **프로세스 ID** | 711937 | 184936 |
| **상태** | ✅ 실행 중 | ⚠️ 실행 중 (API 키 오류) |
| **시작 시간** | 2025-11-24 14:56 (KST) | 2025-11-24 02:00 (KST) |
| **실행 시간** | 약 45분 | 약 13시간 41분 |
| **CPU 사용률** | 1.0% | 1.3% |
| **메모리 사용률** | 0.0% | 2.4% |
| **누적 CPU 시간** | 0:28 | 11:28 |

---

## 📋 작업 1: v4 데이터셋 생성 (generate_data_v4.py)

### 기본 정보

| 항목 | 내용 |
|:---|:---|
| **스크립트 경로** | `/home/pencilfoxs/0_Insurance_PF/05_FineTuning/generate_data_v4.py` |
| **로그 파일** | `generate_data_v4.log` |
| **출력 파일** | `train_dataset_v4_negative_enhanced.json` |
| **API 모델** | Gemini 2.0 Flash Exp |
| **작업 디렉토리** | `/home/pencilfoxs/0_Insurance_PF/05_FineTuning` |

### 진행 상황

| 단계 | 목표 | 현재 진행 | 진행률 | 상태 |
|:---|:---|:---|:---|:---|
| **부정 사례** | 760개 | 536개 | 70.5% | ✅ 진행 중 |
| **긍정 사례** | 1,900개 | 0개 | 0% | ⏳ 대기 중 |
| **계산 시나리오** | 570개 | 0개 | 0% | ⏳ 대기 중 |
| **조항 번호 명시** | 380개 | 0개 | 0% | ⏳ 대기 중 |
| **동의어/검색 실패** | 190개 | 0개 | 0% | ⏳ 대기 중 |
| **전체** | **3,800개** | **536개** | **14.1%** | ✅ 진행 중 |

### 생성 통계

| 항목 | 값 |
|:---|:---|
| **생성된 데이터 수** | 536개 (부정 사례) |
| **파일 크기** | 1.2MB (400개 저장 기준, 실제는 더 많음) |
| **마지막 저장 시간** | 2025-11-24 15:38 (400개 저장) |
| **중간 저장 주기** | 50개마다 자동 저장 ✅ |
| **현재 진행 속도** | 약 2-4초/건 (Rate Limit으로 가변적) |
| **예상 완료 시간** | 약 1.5-2.5시간 (전체 기준) |
| **로그 마지막 업데이트** | 2025-11-24 15:41:22 |

### 유형별 생성 현황

| 유형 | 생성 개수 | 목표 대비 | 상태 |
|:---|:---|:---|:---|
| **negative (부정 사례)** | 536개 | 70.5% (536/760) | ✅ 진행 중 |
| **positive (긍정 사례)** | 0개 | 0% (0/1,900) | ⏳ 대기 중 |
| **calculation (계산)** | 0개 | 0% (0/570) | ⏳ 대기 중 |
| **article (조항)** | 0개 | 0% (0/380) | ⏳ 대기 중 |
| **synonym (동의어)** | 0개 | 0% (0/190) | ⏳ 대기 중 |

### 로그 정보

| 항목 | 내용 |
|:---|:---|
| **로그 파일 경로** | `generate_data_v4.log` |
| **로그 라인 수** | 19줄 |
| **마지막 업데이트** | 실시간 업데이트 중 |
| **모니터링 명령어** | `tail -f generate_data_v4.log` |

---

## 📋 작업 2: Ragas 평가 (evaluate_ragas_full.py)

### 기본 정보

| 항목 | 내용 |
|:---|:---|
| **스크립트 경로** | `/home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation/evaluate_ragas_full.py` |
| **로그 파일** | `nohup_ragas_evaluation.out` |
| **진행 상황 파일** | `results/ragas_evaluation_progress.json` |
| **작업 디렉토리** | `/home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation` |

### 진행 상황

| 항목 | 값 |
|:---|:---|
| **목표 샘플 수** | 200개 (또는 2,223개) |
| **완료된 샘플** | 0개 (진행 상황 파일 기준) |
| **로그 기준 진행률** | 51/200 (26%) |
| **진행률** | 0.0% (진행 상황 파일 기준) / 26% (로그 기준) |
| **상태** | ⚠️ 실행 중 (API 키 만료 오류) |

### 문제점

| 항목 | 내용 |
|:---|:---|
| **오류 유형** | LangSmith API 키 만료 |
| **오류 메시지** | `HTTPError('403 Client Error: Forbidden', '{"error":"API key has expired"}')` |
| **영향** | 로깅 실패 (평가 자체는 계속 진행 중일 수 있음) |
| **해결 방법** | LangSmith API 키 갱신 필요 |

### 리소스 사용

| 항목 | 값 |
|:---|:---|
| **CPU 사용률** | 1.3% |
| **메모리 사용률** | 2.4% (약 2.1GB) |
| **누적 CPU 시간** | 11시간 28분 |

---

## 🔍 상세 모니터링 명령어

### 작업 1: v4 데이터셋 생성

```bash
# 실시간 로그 확인
cd /home/pencilfoxs/0_Insurance_PF/05_FineTuning
tail -f generate_data_v4.log

# 생성된 데이터 확인
python -c "import json; from collections import Counter; data = json.load(open('train_dataset_v4_negative_enhanced.json')); types = Counter([item.get('type', 'unknown') for item in data]); print(f'생성된 데이터: {len(data)}개'); [print(f'  - {k}: {v}개') for k, v in types.items()]"

# 파일 크기 확인
ls -lh train_dataset_v4_negative_enhanced.json

# 프로세스 확인
ps aux | grep generate_data_v4.py | grep -v grep
```

### 작업 2: Ragas 평가

```bash
# 실시간 로그 확인
cd /home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation
tail -f nohup_ragas_evaluation.out

# 진행 상황 확인
cat results/ragas_evaluation_progress.json

# 프로세스 확인
ps aux | grep evaluate_ragas_full.py | grep -v grep
```

---

## ⚠️ 주의사항

1. **SSH 종료:** 두 작업 모두 `nohup`으로 실행되어 SSH 종료 후에도 계속 실행됩니다.
2. **중간 저장:** v4 데이터셋 생성은 50개마다 자동 저장되므로 중단되어도 데이터 손실 없습니다.
3. **API 키 문제:** Ragas 평가 작업의 LangSmith API 키 만료 문제는 평가 자체에는 영향을 주지 않을 수 있으나, 로깅이 실패하고 있습니다.
4. **리소스 모니터링:** 두 작업 모두 장시간 실행되므로 시스템 리소스 모니터링을 권장합니다.

---

## 📈 예상 완료 시간

| 작업 | 예상 완료 시간 | 비고 |
|:---|:---|:---|
| **작업 1 (v4 데이터셋)** | 약 1-2시간 후 | 현재 14.1% 완료 (536/3,800), 부정 사례 70.5% 완료 (536/760) |
| **작업 2 (Ragas 평가)** | 불명확 | API 키 문제로 진행 상황 불확실, 로그 기준 26% (51/200) |

---

## 🔄 다음 업데이트

이 리포트는 주기적으로 업데이트됩니다. 최신 정보는 다음 명령어로 확인하세요:

```bash
cat /home/pencilfoxs/0_Insurance_PF/05_FineTuning/monitoring_report.md
```

