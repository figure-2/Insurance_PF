import json
import os
from collections import defaultdict

# 설정
INPUT_FILE = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset_v4_negative_enhanced.json"
OUTPUT_FILE = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset_v4_clean.json"

def validate_and_clean():
    print(f"🔍 데이터 검증 시작: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ 입력 파일이 존재하지 않습니다.")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 문법 오류 발생: {e}")
        # 파일이 손상되었을 경우 복구 로직이 필요할 수 있음 (여기선 생략)
        return

    print(f"📥 로드된 데이터: {len(data)}개")
    
    cleaned_data = []
    seen_hashes = set()
    stats = defaultdict(int)
    
    for item in data:
        # 1. 필수 필드 체크
        if not all(k in item for k in ['instruction', 'input', 'output', 'type']):
            stats['missing_fields'] += 1
            continue
            
        # 2. 내용 유효성 체크 (너무 짧거나 비어있는 경우)
        if len(item['output'].strip()) < 5:
            stats['too_short'] += 1
            continue
            
        # 3. 중복 제거 (Instruction + Input 기준)
        # 띄어쓰기 무시하고 비교
        content_hash = hash((item['instruction'].replace(" ", ""), item['input'].replace(" ", "")))
        if content_hash in seen_hashes:
            stats['duplicate'] += 1
            continue
        
        seen_hashes.add(content_hash)
        
        # 4. Type 정규화 (혹시 모를 오타 방지)
        q_type = item['type'].lower().strip()
        item['type'] = q_type
        
        cleaned_data.append(item)
        stats[f'type_{q_type}'] += 1

    # 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("📊 검증 결과 리포트")
    print("="*50)
    print(f"✅ 유효 데이터: {len(cleaned_data)}개 (저장됨: {OUTPUT_FILE})")
    print(f"🗑️ 제거된 데이터:")
    print(f"   - 중복: {stats['duplicate']}개")
    print(f"   - 필드 누락: {stats['missing_fields']}개")
    print(f"   - 내용 부실: {stats['too_short']}개")
    print("\n📈 유형별 분포 (Cleaned):")
    for k, v in stats.items():
        if k.startswith('type_'):
            print(f"   - {k.replace('type_', '').upper()}: {v}개")
    print("="*50)

if __name__ == "__main__":
    validate_and_clean()
