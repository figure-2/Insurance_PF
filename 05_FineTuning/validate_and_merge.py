import json
import os
import glob
from pathlib import Path

def validate_and_merge(input_dir, output_file):
    print(f"🔍 데이터 통합 및 검증 시작: {input_dir}")
    
    all_files = sorted(glob.glob(os.path.join(input_dir, "dataset_part_*.json")))
    print(f"📂 발견된 파일: {len(all_files)}개")
    
    merged_data = []
    seen_instructions = set()
    stats = {
        "total_read": 0,
        "valid": 0,
        "filtered_short": 0,    # 너무 짧음
        "filtered_incomplete": 0, # 문장 안 끝남
        "filtered_duplicate": 0,  # 중복 질문
        "filtered_no_answer": 0,  # "없습니다" 류
        "types": {"fact": 0, "scenario": 0, "easy": 0, "unknown": 0}
    }
    
    # "없습니다" 류의 무의미한 답변 패턴
    invalid_patterns = [
        "명시되어 있지 않습니다",
        "해당 내용이 없습니다",
        "알 수 없습니다",
        "제공된 약관에는",
        "언급되어 있지 않습니다"
    ]

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats["total_read"] += len(data)
                
                for item in data:
                    instruction = item.get('instruction', '').strip()
                    output = item.get('output', '').strip()
                    q_type = item.get('type', 'unknown')
                    
                    # 1. 필수 필드 체크
                    if not instruction or not output:
                        continue
                        
                    # 2. 중복 체크
                    if instruction in seen_instructions:
                        stats["filtered_duplicate"] += 1
                        continue
                    
                    # 3. 길이 체크 (너무 짧은 답변 제외)
                    if len(output) < 10:
                        stats["filtered_short"] += 1
                        continue
                        
                    # 4. 완결성 체크 (문장이 잘렸는지 확인)
                    # 한글/영문 문장 부호로 끝나는지 확인
                    if not output[-1] in ['.', '!', '?', '"', "'", '다', '요', '죠']:
                        stats["filtered_incomplete"] += 1
                        continue
                        
                    # 5. 유효성 체크 (무의미한 답변 제외)
                    # output이 invalid_patterns 중 하나를 포함하고, 길이가 짧으면(예: 50자 미만) 제외
                    is_invalid = False
                    if len(output) < 60:
                        for pattern in invalid_patterns:
                            if pattern in output:
                                is_invalid = True
                                break
                    
                    if is_invalid:
                        stats["filtered_no_answer"] += 1
                        continue

                    # 통과
                    seen_instructions.add(instruction)
                    merged_data.append(item)
                    stats["valid"] += 1
                    stats["types"][q_type] = stats["types"].get(q_type, 0) + 1
                    
        except Exception as e:
            print(f"❌ 파일 읽기 오류 ({file_path}): {e}")

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*50)
    print("📊 [검증 및 통합 결과]")
    print(f"총 읽은 데이터: {stats['total_read']:,}개")
    print(f"✅ 최종 유효 데이터: {stats['valid']:,}개")
    print("-" * 30)
    print(f"🗑️ [필터링 내역]")
    print(f"  - 중복 질문 제거: {stats['filtered_duplicate']:,}개")
    print(f"  - 너무 짧은 답변: {stats['filtered_short']:,}개")
    print(f"  - 문장 불완전(잘림): {stats['filtered_incomplete']:,}개")
    print(f"  - 무의미한 답변(없음 등): {stats['filtered_no_answer']:,}개")
    print("-" * 30)
    print(f"📈 [유형별 분포]")
    print(f"  - Fact: {stats['types']['fact']:,}")
    print(f"  - Scenario: {stats['types']['scenario']:,}")
    print(f"  - Easy: {stats['types']['easy']:,}")
    print("="*50)
    print(f"💾 저장 완료: {output_file}")

if __name__ == "__main__":
    # 경로 설정
    BASE_DIR = Path("/home/pencilfoxs/0_Insurance_PF/05_FineTuning")
    INPUT_DIR = BASE_DIR / "generated_data_v2"
    OUTPUT_FILE = BASE_DIR / "train_dataset_final_v2.json"
    
    validate_and_merge(INPUT_DIR, OUTPUT_FILE)
