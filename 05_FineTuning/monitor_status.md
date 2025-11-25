# 백그라운드 작업 모니터링 상태

**최종 업데이트:** 2025-11-24 15:10 (KST)

---

## 작업 1: v4 데이터셋 생성 (generate_data_v4.py)

**프로세스 ID:** 711937  
**시작 시간:** 2025-11-24 14:56 (KST)  
**실행 시간:** 약 9분 경과  
**상태:** ✅ 실행 중

### 진행 상황
- **현재 단계:** 부정 사례 생성 중
- **진행률:** 98/760 (약 12.9%)
- **생성 속도:** 약 2-4초/건 (Rate Limit으로 인해 가변적)
- **예상 완료 시간:** 약 1.5-2.5시간 (전체 3,800개 기준)

### 생성 계획
- 부정 사례: 760개 (20%)
- 긍정 사례: 1,900개 (50%)
- 계산 시나리오: 570개 (15%)
- 조항 번호 명시: 380개 (10%)
- 동의어/검색 실패: 190개 (5%)
- **총계:** 3,800개

### 현재 생성된 데이터
- **파일:** `train_dataset_v4_negative_enhanced.json`
- **생성된 데이터:** 50개 (부정 사례, 50개마다 중간 저장)
- **중간 저장:** 50개마다 자동 저장 ✅
- **최종 저장:** 모든 생성 완료 후

### 로그 파일
- **경로:** `/home/pencilfoxs/0_Insurance_PF/05_FineTuning/generate_data_v4.log`
- **모니터링:** `tail -f generate_data_v4.log`

---

## 작업 2: Ragas 평가 (evaluate_ragas_full.py)

**프로세스 ID:** 184936  
**시작 시간:** 2025-11-24 02:00 (KST)  
**실행 시간:** 약 13시간 경과  
**상태:** ✅ 실행 중

### 진행 상황
- **작업 경로:** `/home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation`
- **로그 파일:** `nohup_ragas_evaluation.out`
- **진행 상황 파일:** `results/ragas_evaluation_progress.json`
- **진행률:** 51/200 (26%)
- **⚠️ 문제:** LangSmith API 키 만료 오류 발생 (평가는 계속 진행 중이나 로깅 실패)

### 모니터링 명령어
```bash
# 로그 확인
cd /home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation
tail -f nohup_ragas_evaluation.out

# 진행 상황 확인
cat results/ragas_evaluation_progress.json
```

---

## 모니터링 명령어 요약

### 작업 1 (v4 데이터셋 생성)
```bash
# 로그 확인
cd /home/pencilfoxs/0_Insurance_PF/05_FineTuning
tail -f generate_data_v4.log

# 생성된 데이터 확인
python -c "import json; data = json.load(open('train_dataset_v4_negative_enhanced.json')); print(f'생성된 데이터: {len(data)}개')"

# 프로세스 확인
ps aux | grep generate_data_v4.py | grep -v grep
```

### 작업 2 (Ragas 평가)
```bash
# 로그 확인
cd /home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation
tail -f nohup_ragas_evaluation.out

# 진행 상황 확인
cat results/ragas_evaluation_progress.json

# 프로세스 확인
ps aux | grep evaluate_ragas_full.py | grep -v grep
```

---

## 주의사항

1. **SSH 종료:** 두 작업 모두 `nohup`으로 실행되어 SSH 종료 후에도 계속 실행됩니다.
2. **중간 저장:** v4 데이터셋 생성은 50개마다 자동 저장되므로 중단되어도 데이터 손실 없습니다.
3. **리소스 사용:** 두 작업 모두 CPU/메모리를 사용하므로 시스템 리소스 모니터링 권장.

