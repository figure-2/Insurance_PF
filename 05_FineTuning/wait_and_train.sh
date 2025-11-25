#!/bin/bash
# 데이터 생성 완료 대기 후 자동으로 파인튜닝 실행

SCRIPT_DIR="/home/pencilfoxs/0_Insurance_PF/05_FineTuning"
DATASET_FILE="$SCRIPT_DIR/train_dataset.json"
LOG_FILE="$SCRIPT_DIR/generate_data_full.log"

echo "=========================================="
echo "데이터 생성 완료 대기 중..."
echo "=========================================="

# 데이터 생성 프로세스가 실행 중인지 확인
while ps aux | grep -v grep | grep -q "generate_data.py"; do
    if [ -f "$LOG_FILE" ]; then
        PROGRESS=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -oP '\d+/\d+' | head -1)
        if [ ! -z "$PROGRESS" ]; then
            echo "[$(date '+%H:%M:%S')] 진행 중: $PROGRESS"
        fi
    fi
    sleep 60  # 1분마다 확인
done

echo ""
echo "=========================================="
echo "데이터 생성 완료 확인 중..."
echo "=========================================="

# 데이터셋 파일 존재 확인
if [ -f "$DATASET_FILE" ]; then
    SAMPLE_COUNT=$(python3 -c "import json; data = json.load(open('$DATASET_FILE')); print(len(data))" 2>/dev/null)
    echo "✅ 데이터셋 생성 완료: $SAMPLE_COUNT 개 샘플"
    echo ""
    echo "=========================================="
    echo "Step 4: QLoRA 파인튜닝 시작..."
    echo "=========================================="
    
    cd "$SCRIPT_DIR"
    python train_qlora.py 2>&1 | tee train_qlora.log
    
    echo ""
    echo "=========================================="
    echo "파인튜닝 완료!"
    echo "=========================================="
else
    echo "❌ 오류: 데이터셋 파일을 찾을 수 없습니다."
    exit 1
fi

