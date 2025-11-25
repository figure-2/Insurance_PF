import json
import random
import os
from collections import defaultdict

# 설정
INPUT_FILE = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset_v4_clean.json"
TRAIN_FILE = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_v4.json"
TEST_FILE = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/test_v4.json"
SPLIT_RATIO = 0.8  # 80% Train, 20% Test
SEED = 42

def stratified_split():
    print(f"🔪 데이터 분할 시작 (비율 {SPLIT_RATIO}:{1-SPLIT_RATIO:.1f})")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 유형별로 그룹화
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item['type']].append(item)
    
    train_data = []
    test_data = []
    
    random.seed(SEED)
    
    print("\n📊 유형별 분할 현황:")
    print(f"{'TYPE':<15} | {'TOTAL':<8} | {'TRAIN':<8} | {'TEST':<8}")
    print("-" * 50)
    
    for q_type, items in grouped_data.items():
        random.shuffle(items) # 무작위 섞기
        
        split_idx = int(len(items) * SPLIT_RATIO)
        
        train_chunk = items[:split_idx]
        test_chunk = items[split_idx:]
        
        train_data.extend(train_chunk)
        test_data.extend(test_chunk)
        
        print(f"{q_type.upper():<15} | {len(items):<8} | {len(train_chunk):<8} | {len(test_chunk):<8}")
        
    # 최종 데이터도 한 번 더 섞기 (학습 시 편향 방지)
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 저장
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print("-" * 50)
    print(f"✅ 저장 완료:")
    print(f"   📁 학습용(Train): {TRAIN_FILE} ({len(train_data)}개)")
    print(f"   📁 평가용(Test) : {TEST_FILE} ({len(test_data)}개)")
    print("=" * 50)

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 먼저 검증 스크립트(validate_and_merge_v4.py)를 실행하여 clean 파일을 생성해주세요.")
    else:
        stratified_split()
