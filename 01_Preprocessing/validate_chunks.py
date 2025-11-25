import json
import os
import random
import re

def validate_chunks(file_path):
    """
    청킹된 JSONL 파일을 검증하고 리포트를 출력합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Validation Start: {os.path.basename(file_path)} ---")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
        
    total_chunks = len(chunks)
    print(f"Total Chunks: {total_chunks}")
    
    if total_chunks == 0:
        print("Error: No chunks found.")
        return

    # --- 1. 필수 메타데이터 검증 ---
    missing_meta = []
    missing_breadcrumbs = []
    
    for i, c in enumerate(chunks):
        meta = c.get('metadata', {})
        if not meta:
            missing_meta.append(i)
        if not meta.get('breadcrumbs'):
            missing_breadcrumbs.append(i)
            
    print(f"\n1. Metadata Integrity:")
    print(f"   - Missing Metadata: {len(missing_meta)} chunks")
    print(f"   - Missing Breadcrumbs: {len(missing_breadcrumbs)} chunks")
    if missing_breadcrumbs:
        print(f"     Example missing breadcrumb chunk ID: {chunks[missing_breadcrumbs[0]]['chunk_id']}")

    # --- 2. 노이즈(헤더/푸터) 잔존 여부 확인 ---
    # 삭제했어야 할 키워드들이 본문에 남아있는지 확인
    noise_keywords = ["보통약관", "특별약관", "개인용자동차보험", "알기쉬운 자동차보험 이야기", "관련 법령"]
    noise_counts = {k: 0 for k in noise_keywords}
    
    # 주의: '보통약관'이라는 단어 자체가 본문 내용일 수도 있으므로, 
    # 여기서는 "헤더처럼 단독으로 존재하는 라인"이 있는지 체크하는 것이 더 정확함
    # 하지만 간단히 포함 여부만 먼저 봄.
    
    for c in chunks:
        text = c['text']
        for k in noise_keywords:
            # 단순히 포함된 것(in)은 본문 내용일 수 있음.
            # 줄 단위로 쪼개서, 해당 줄이 키워드와 정확히 일치하거나 앞뒤 공백만 있는 경우를 노이즈로 간주
            lines = text.split('\n')
            for line in lines:
                if line.strip() == k:
                    noise_counts[k] += 1
                    break # 한 청크에 하나만 있어도 카운트
                    
    print(f"\n2. Noise Residue Check (Exact Line Match):")
    for k, v in noise_counts.items():
        print(f"   - '{k}': detected in {v} chunks ({v/total_chunks*100:.1f}%)")


    # --- 3. 표(Markdown Table) 변환 확인 ---
    # | --- | 패턴이 있는지 확인
    table_pattern = re.compile(r'\|.*\|')
    chunks_with_table_text = 0
    
    for c in chunks:
        if table_pattern.search(c['text']):
            chunks_with_table_text += 1
            
    print(f"\n3. Markdown Table Check:")
    print(f"   - Chunks with Markdown Table syntax: {chunks_with_table_text} ({chunks_with_table_text/total_chunks*100:.1f}%)")


    # --- 4. 청크 길이 분포 ---
    lengths = [len(c['text']) for c in chunks]
    avg_len = sum(lengths) / len(lengths)
    
    print(f"\n4. Text Length Statistics (Characters):")
    print(f"   - Average: {avg_len:.1f}")
    print(f"   - Min: {min(lengths)}")
    print(f"   - Max: {max(lengths)}")
    
    short_chunks = [c for c in chunks if len(c['text']) < 50]
    if short_chunks:
        print(f"   - Warning: {len(short_chunks)} chunks are shorter than 50 chars.")
        print(f"     Sample short chunk: {repr(short_chunks[0]['text'])}")


    # --- 5. 랜덤 샘플링 (정성 평가용) ---
    print(f"\n5. Random Sample (Human Inspection):")
    sample = random.choice(chunks)
    print("-" * 40)
    print(f"Chunk ID: {sample['chunk_id']}")
    print(f"Breadcrumbs: {sample['metadata'].get('breadcrumbs')}")
    print("-" * 40)
    print(sample['text'][:500] + ("..." if len(sample['text']) > 500 else ""))
    print("-" * 40)
    
    print("\n--- Validation Complete ---")

if __name__ == "__main__":
    target_file = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
    validate_chunks(target_file)

