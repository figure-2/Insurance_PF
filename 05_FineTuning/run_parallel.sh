#!/bin/bash

# 작업 디렉토리 설정
WORK_DIR="/home/pencilfoxs/0_Insurance_PF/05_FineTuning"
cd "$WORK_DIR" || exit 1

# 총 사용할 키 개수 (2번부터 9번까지 8개 사용 예정)
TOTAL_KEYS=8

# 사용할 키 번호 시작점 (예: GOOGLE_API_KEY_2 부터 시작하면 START_KEY=2)
START_KEY=2

echo "🚀 Starting Parallel Data Generation with $TOTAL_KEYS keys..."
echo "📁 Working Directory: $WORK_DIR"

for ((i=0; i<TOTAL_KEYS; i++)); do
    KEY_NUM=$((START_KEY + i))
    
    # 백그라운드 실행 (nohup으로 SSH 종료 후에도 계속 실행)
    # 로그는 logs 폴더에 별도 저장
    mkdir -p logs
    nohup python3 "$WORK_DIR/generate_data_parallel.py" \
        --key_num $KEY_NUM \
        --total_keys $TOTAL_KEYS \
        --output_dir generated_data_v2 \
        > "$WORK_DIR/logs/worker_$KEY_NUM.log" 2>&1 &
        
    echo "   ✅ Started Worker $KEY_NUM (PID $!)"
    sleep 1 # 프로세스 생성 간격
done

echo ""
echo "✅ All workers started!"
echo "📊 Monitor logs: tail -f $WORK_DIR/logs/worker_*.log"
echo "📊 Check processes: ps aux | grep generate_data_parallel"
