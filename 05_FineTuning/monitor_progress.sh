#!/bin/bash

# 데이터 생성 진행 상황 모니터링 스크립트

WORK_DIR="/home/pencilfoxs/0_Insurance_PF/05_FineTuning"
cd "$WORK_DIR" || exit 1

clear
echo "=========================================="
echo "📊 데이터 생성 진행 상황 모니터링"
echo "=========================================="
echo "⏰ 현재 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 실행 중인 워커 확인
echo "=== 🔄 실행 중인 워커 ==="
worker_count=$(ps aux | grep "generate_data_parallel" | grep -v grep | wc -l)
if [ $worker_count -eq 0 ]; then
    echo "  ⚠️  실행 중인 워커가 없습니다!"
else
    echo "  ✅ 총 $worker_count 개 워커 실행 중"
    ps aux | grep "generate_data_parallel" | grep -v grep | awk '{printf "    Worker PID: %s | CPU: %s%% | MEM: %s%% | 실행시간: %s\n", $2, $3, $4, $10}'
fi
echo ""

# 2. 각 워커별 진행 상황
echo "=== 📈 각 워커별 진행 상황 ==="
total_processed=0
for i in {2..8}; do
    if [ -f "logs/worker_$i.log" ]; then
        # 진행률 추출 (예: "3/801")
        progress=$(tail -1 "logs/worker_$i.log" 2>/dev/null | grep -oP '\d+/\d+' | head -1)
        if [ -n "$progress" ]; then
            current=$(echo $progress | cut -d'/' -f1)
            total=$(echo $progress | cut -d'/' -f2)
            if [ -n "$current" ] && [ -n "$total" ]; then
                percent=$(echo "scale=1; $current * 100 / $total" | bc 2>/dev/null || echo "0")
                echo "  Worker $i: $progress ($percent%)"
                total_processed=$((total_processed + current))
            fi
        fi
    fi
done
echo ""

# 3. 생성된 데이터 파일
echo "=== 📁 생성된 데이터 파일 ==="
if [ -d "generated_data_v2" ]; then
    file_count=$(find generated_data_v2 -name "dataset_part_*.json" -type f 2>/dev/null | wc -l)
    if [ $file_count -gt 0 ]; then
        echo "  ✅ $file_count 개 파일 생성됨"
        total_samples=0
        for f in generated_data_v2/dataset_part_*.json; do
            if [ -f "$f" ]; then
                count=$(python3 -c "import json; f=open('$f'); data=json.load(f); print(len(data))" 2>/dev/null || echo "0")
                size=$(ls -lh "$f" | awk '{print $5}')
                echo "    $(basename $f): $count 개 샘플 ($size)"
                total_samples=$((total_samples + count))
            fi
        done
        echo "  📊 총 생성된 QA 쌍: $total_samples 개"
        estimated_chunks=$((total_samples / 3))
        total_chunks=6402
        if [ $total_chunks -gt 0 ]; then
            progress_pct=$(echo "scale=2; $estimated_chunks * 100 / $total_chunks" | bc 2>/dev/null || echo "0")
            echo "  📈 예상 진행률: 약 ${progress_pct}% ($estimated_chunks / $total_chunks 청크)"
        fi
    else
        echo "  ⏳ 아직 파일 생성 전... (10개 청크마다 저장)"
    fi
else
    echo "  ⏳ 출력 디렉토리 아직 생성 안 됨"
fi
echo ""

# 4. 에러 확인
echo "=== ⚠️  에러 확인 ==="
error_found=0
for i in {2..8}; do
    if [ -f "logs/worker_$i.log" ]; then
        if grep -qi "error\|exception\|failed\|❌" "logs/worker_$i.log" 2>/dev/null; then
            echo "  ⚠️  Worker $i 에러 발견:"
            grep -i "error\|exception\|failed\|❌" "logs/worker_$i.log" 2>/dev/null | tail -2 | sed 's/^/    /'
            error_found=1
        fi
    fi
done
if [ $error_found -eq 0 ]; then
    echo "  ✅ 에러 없음"
fi
echo ""

# 5. 예상 완료 시간
echo "=== ⏱️  예상 완료 시간 ==="
if [ $total_processed -gt 0 ]; then
    # 평균 속도 계산 (초당 처리 청크 수)
    # 각 워커가 약 25초/청크 소요 (로그에서 확인)
    avg_time_per_chunk=25
    remaining_chunks=$((6402 - total_processed))
    total_seconds=$((remaining_chunks * avg_time_per_chunk / 7))  # 7개 워커로 병렬 처리
    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    echo "  예상 남은 시간: 약 ${hours}시간 ${minutes}분"
    echo "  (현재 속도 기준, 실제 속도는 변동 가능)"
fi
echo ""

echo "=========================================="
echo "💡 모니터링 명령어:"
echo "   tail -f $WORK_DIR/logs/worker_*.log"
echo "   ps aux | grep generate_data_parallel"
echo "=========================================="
