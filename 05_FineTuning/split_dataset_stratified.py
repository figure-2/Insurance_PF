import json
import os
import random
from collections import defaultdict
from pathlib import Path

# 경로 설정
BASE_DIR = Path("/home/pencilfoxs/0_Insurance_PF/05_FineTuning")
INPUT_FILE = BASE_DIR / "train_dataset_final_v2.json"
TRAIN_OUTPUT = BASE_DIR / "train_80.json"
TEST_OUTPUT = BASE_DIR / "test_20.json"

def split_dataset():
    print(f"📂 데이터 로드 중: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_count = len(data)
    print(f"📊 전체 데이터: {total_count}개")
    
    # 유형별 그룹화
    grouped_data = defaultdict(list)
    for item in data:
        q_type = item.get('type', 'unknown')
        grouped_data[q_type].append(item)
    
    train_data = []
    test_data = []
    
    print("\n🔍 유형별 분할 (Train: 80% / Test: 20%)")
    print("-" * 40)
    print(f"{'Type':<15} {'Total':<10} {'Train':<10} {'Test':<10}")
    print("-" * 40)
    
    # 랜덤 시드 고정 (재현성 확보)
    random.seed(42)
    
    for q_type, items in grouped_data.items():
        # 각 유형 내에서 셔플
        random.shuffle(items)
        
        # 8:2 지점 계산
        split_idx = int(len(items) * 0.8)
        
        train_chunk = items[:split_idx]
        test_chunk = items[split_idx:]
        
        train_data.extend(train_chunk)
        test_data.extend(test_chunk)
        
        print(f"{q_type:<15} {len(items):<10} {len(train_chunk):<10} {len(test_chunk):<10}")
        
    print("-" * 40)
    print(f"{'Total':<15} {total_count:<10} {len(train_data):<10} {len(test_data):<10}")
    
    # 다시 전체 셔플 (학습 시 데이터 분포가 섞이도록)
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 저장
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(TEST_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 분할 및 저장 완료!")
    print(f"  👉 학습용: {TRAIN_OUTPUT}")
    print(f"  👉 평가용: {TEST_OUTPUT}")

if __name__ == "__main__":
    split_dataset()
