import json
import random
import argparse
import os
import time
import requests
import math
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 환경 설정 및 API 키 로드
# -----------------------------------------------------------------------------
def load_api_key(key_num):
    """
    .env2 파일에서 지정된 번호의 API 키를 로드합니다.
    예: key_num=2 -> GOOGLE_API_KEY_2
    """
    env_paths = ['/home/pencilfoxs/00_new/.env2', '/home/pencilfoxs/PJ/.env2']
    
    # 환경 변수 이름 결정
    target_key_name = f"GOOGLE_API_KEY_{key_num}" if key_num > 1 else "GOOGLE_API_KEY"
    
    # 1. os.environ 확인
    for path in env_paths:
        if os.path.exists(path):
            load_dotenv(path)
            
    api_key = os.getenv(target_key_name)
    
    # 2. 파일 직접 파싱 (fallback)
    if not api_key:
        for path in env_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 주석 제거 및 공백 제거
                        clean_line = line.split('#')[0].strip()
                        if clean_line.startswith(f"{target_key_name}="):
                            api_key = clean_line.split('=', 1)[1].strip()
                            break
            if api_key: break
            
    if not api_key:
        raise ValueError(f"API Key {target_key_name} not found in .env2 files.")
        
    return api_key

# 모델명 (fallback 지원)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# -----------------------------------------------------------------------------
# 2. Multi-Persona Prompts (벤치마킹 적용)
# -----------------------------------------------------------------------------
PROMPT_FACT = """
당신은 **'약관을 꼼꼼히 따지는 깐깐한 보험 가입자'**입니다.
제공된 [보험 약관]을 읽고, 보상 한도, 면책 기간, 지급 조건 등 **정확한 사실이나 수치**를 확인하는 날카로운 질문을 하나만 만드세요.

[규칙]
1. 모호한 질문 금지. 정확한 숫자나 조건을 물어보세요.
2. 예: "암 진단비의 감액 기간은 가입 후 정확히 며칠인가요?"
3. 반드시 질문 하나만 만드세요.

[보험 약관]
{text}

[생성 형식]
질문: [질문내용]
답변: [약관에 근거한 명확한 답변]
"""

PROMPT_SCENARIO = """
당신은 **'갑작스러운 사고를 당해 당황한 피해자'**입니다.
제공된 [보험 약관]과 관련된 **구체적인 사고 상황(시나리오)**을 가정하고, 이 경우 보상이 가능한지 묻는 질문을 하나만 만드세요.

[규칙]
1. "제가 ~한 상황인데요," 처럼 구체적인 상황을 묘사하세요.
2. 예: "주차장에서 문을 열다가 옆 차를 살짝 긁었는데, 이것도 보상되나요?"
3. 반드시 질문 하나만 만드세요.

[보험 약관]
{text}

[생성 형식]
질문: [질문내용]
답변: [약관을 적용하여 상황에 맞게 설명한 답변]
"""

PROMPT_EASY = """
당신은 **'보험 용어를 전혀 모르는 사회 초년생'**입니다.
제공된 [보험 약관]의 내용을 묻되, 전문 용어(예: 기왕증, 면책금)를 쓰지 말고 **쉬운 말로 풀어서** 질문하세요.

[규칙]
1. 전문 용어 대신 "그거 있잖아요", "내가 내야 하는 돈" 같은 표현을 쓰세요.
2. 예: "자기부담금이 뭔가요?" (X) -> "사고 나면 제가 쌩돈으로 내야 하는 금액이 얼마인가요?" (O)
3. 반드시 질문 하나만 만드세요.

[보험 약관]
{text}

[생성 형식]
질문: [질문내용]
답변: [초보자도 이해하기 쉽게 풀어서 쓴 답변]
"""

# -----------------------------------------------------------------------------
# 3. Robust API Call (지수 백오프 적용 + Rate Limit 대응 강화)
# -----------------------------------------------------------------------------
def call_gemini_api_robust(api_key, payload, max_retries=5):
    headers = {"Content-Type": "application/json"}
    base_delay = 2.0
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            
            # 429(Rate Limit) -> 더 긴 대기 시간
            if response.status_code == 429:
                # Rate Limit의 경우 더 긴 대기 시간 (최대 30초로 단축)
                wait_time = min(base_delay * (2 ** attempt) * 2 + random.uniform(1, 3), 30)
                if attempt < max_retries:
                    print(f"⚠️ Rate Limit (429) 발생. {wait_time:.1f}초 대기 후 재시도...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate Limit (429) 재시도 실패. None 반환")
                    return None
            
            # 500(Server Error) -> Retry
            if response.status_code >= 500:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue
            
            # 400(Bad Request) 등은 즉시 실패 처리
            return None
            
        except Exception as e:
            wait_time = base_delay * (2 ** attempt)
            if attempt < max_retries:
                time.sleep(wait_time)
    
    return None

def parse_response(response_json, q_type, chunk_text, chunk_id):
    try:
        if 'candidates' not in response_json or not response_json['candidates']:
            return None
            
        candidate = response_json['candidates'][0]
        if 'content' not in candidate or 'parts' not in candidate['content']:
            return None
            
        content = candidate['content']['parts'][0]['text']
        
        lines = content.strip().split('\n')
        question = ""
        answer = ""
        
        for line in lines:
            if line.startswith("질문:"):
                question = line.replace("질문:", "").strip()
            elif line.startswith("답변:"):
                answer = line.replace("답변:", "").strip()
        
        if not question or not answer:
            return None
            
        return {
            "chunk_id": chunk_id,
            "instruction": question,
            "input": chunk_text,
            "output": answer,
            "type": q_type
        }
    except:
        return None

# -----------------------------------------------------------------------------
# 4. Main Logic
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_num", type=int, required=True, help="API Key Number (1-20)")
    parser.add_argument("--total_keys", type=int, default=1, help="Total number of keys used")
    parser.add_argument("--input", default="/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl")
    parser.add_argument("--output_dir", default="generated_data")
    args = parser.parse_args()

    # 1. API 키 로드
    try:
        api_key = load_api_key(args.key_num)
        print(f"✅ [Worker {args.key_num}] API Key Loaded")
    except Exception as e:
        print(f"❌ [Worker {args.key_num}] Error: {e}")
        return

    # 2. 청크 로드
    with open(args.input, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    # 3. 데이터 분할 (Sharding)
    # 전체 데이터를 total_keys로 나누어 내 몫만 가져옴
    total_chunks = len(all_chunks)
    chunk_size = math.ceil(total_chunks / args.total_keys)
    start_idx = (args.key_num - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total_chunks)
    
    my_chunks = all_chunks[start_idx:end_idx]
    print(f"📊 [Worker {args.key_num}] Assigned: {start_idx} ~ {end_idx-1} ({len(my_chunks)} chunks)")
    
    if not my_chunks:
        print(f"✅ [Worker {args.key_num}] No chunks assigned. Exiting.")
        return

    # 4. 출력 파일 설정 및 Resume 준비
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"dataset_part_{args.key_num}.json")
    
    processed_chunk_ids = set()
    dataset = []
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                # 이미 처리된 chunk_id 수집 (한 청크에 3개 질문이므로 chunk_id로 체크)
                for item in dataset:
                    if 'chunk_id' in item:
                        processed_chunk_ids.add(item['chunk_id'])
            print(f"📂 [Worker {args.key_num}] Resuming: {len(processed_chunk_ids)} chunks already processed")
        except:
            print(f"⚠️ [Worker {args.key_num}] Output file load failed. Starting fresh.")
            dataset = []

    # 5. 생성 루프
    prompts = [('fact', PROMPT_FACT), ('scenario', PROMPT_SCENARIO), ('easy', PROMPT_EASY)]
    
    # 이미 처리된 청크는 건너뜀
    target_chunks = [c for c in my_chunks if c['chunk_id'] not in processed_chunk_ids]
    
    for i, chunk in enumerate(tqdm(target_chunks, desc=f"Worker {args.key_num}", position=args.key_num)):
        chunk_text = chunk['text']
        chunk_id = chunk['chunk_id']
        
        chunk_results = []
        for p_type, prompt_tmpl in prompts:
            payload = {
                "contents": [{"parts": [{"text": prompt_tmpl.format(text=chunk_text[:1500])}]}],
                "generationConfig": {
                    "temperature": 0.7 if p_type == 'scenario' else 0.4,
                    "maxOutputTokens": 2000
                }
            }
            
            # Rate Limit 회피를 위한 요청 간 간격 추가 (0.2초로 단축)
            time.sleep(0.2)
            
            result = call_gemini_api_robust(api_key, payload)
            if result:
                parsed = parse_response(result, p_type, chunk_text, chunk_id)
                if parsed:
                    chunk_results.append(parsed)
        
        # 결과 저장 (하나라도 성공했으면)
        if chunk_results:
            dataset.extend(chunk_results)
            
        # 10개 청크마다 파일 저장 (데이터 손실 방지)
        if (i + 1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

    # 최종 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"✅ [Worker {args.key_num}] Completed! Total {len(dataset)} QA pairs generated.")

if __name__ == "__main__":
    main()
