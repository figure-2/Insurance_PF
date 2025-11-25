"""
파인튜닝용 QA 데이터셋 생성 스크립트 (GEMINI 2.5 pro 사용)
약관 청크를 Context로 포함하여 "문서를 보고 답변하는" 학습 데이터 생성
GEMINI 2.5 pro의 우수한 한국어 성능과 환각 감소 기능 활용
"""

import json
import random
import argparse
import os
import time
import requests
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드 (여러 경로 시도)
env_paths = [
    '/home/pencilfoxs/00_new/.env2',  # 우선순위 1
    '/home/pencilfoxs/PJ/.env2'        # 우선순위 2
]

env_path = None
for path in env_paths:
    if os.path.exists(path):
        env_path = path
        load_dotenv(path)
        break

if not env_path:
    print(f"⚠️  환경 변수 파일을 찾을 수 없습니다: {env_paths}")

# 설정
DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
OUTPUT_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset_rag_gemini.json"

# API 키 찾기 (우선순위: GOOGLE_API_KEY_19 > GOOGLE_API_KEY_2 > GOOGLE_API_KEY > GOOGLE_AI_STUDIO_API_KEY)
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY_19') or os.getenv('GOOGLE_API_KEY_2') or os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_AI_STUDIO_API_KEY')

# API 키가 없으면 직접 파일에서 읽기 시도
if not GEMINI_API_KEY and os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            # 주석이나 빈 줄 건너뛰기
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            # 우선순위에 따라 키 찾기
            for key_name in ['GOOGLE_API_KEY_2', 'GOOGLE_API_KEY', 'GOOGLE_AI_STUDIO_API_KEY']:
                if key_name in line_stripped and '=' in line_stripped:
                    parts = line_stripped.split('=', 1)
                    if len(parts) == 2:
                        GEMINI_API_KEY = parts[1].strip()
                        # 주석 제거
                        if '#' in GEMINI_API_KEY:
                            GEMINI_API_KEY = GEMINI_API_KEY.split('#')[0].strip()
                        print(f"✅ {key_name} 사용")
                        break
            if GEMINI_API_KEY:
                break

if not GEMINI_API_KEY:
    raise ValueError("GEMINI API 키를 찾을 수 없습니다. .env2 파일에 GOOGLE_API_KEY_2, GOOGLE_API_KEY, 또는 GOOGLE_AI_STUDIO_API_KEY를 설정하세요.")

print(f"✅ API 키 로드 완료 (길이: {len(GEMINI_API_KEY)}자)")


def init_gemini():
    """GEMINI 모델 초기화 (SDK 우선, REST API 폴백)"""
    print("Initializing GEMINI...")
    
    # SDK 방식 먼저 시도 (더 안정적)
    try:
        model = init_gemini_sdk()
        print("✅ GEMINI SDK 초기화 성공")
        return model
    except Exception as e_sdk:
        print(f"⚠️  SDK 방식 실패: {e_sdk}")
        print("REST API 방식으로 재시도...")
        
        # REST API 엔드포인트 설정 (v1beta 사용, gemini-2.5-flash 사용)
        model_name = 'gemini-2.5-flash'
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        
        print(f"✅ Using model: {model_name} (REST API)")
        
        # 테스트 요청
        test_payload = {
            "contents": [{
                "parts": [{"text": "테스트"}]
            }]
        }
        
        try:
            response = requests.post(api_url, json=test_payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                # 응답 구조 확인
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        print("✅ GEMINI API 연결 성공 (REST API 사용)")
                        return {'api_url': api_url, 'api_key': GEMINI_API_KEY}
                    else:
                        print(f"⚠️  REST API 응답 구조 문제: {list(candidate.keys())}")
                        raise ValueError("REST API 응답 구조가 예상과 다릅니다.")
                else:
                    raise ValueError("REST API 응답에 candidates가 없습니다.")
            else:
                print(f"⚠️  API 응답 코드: {response.status_code}")
                print(f"   응답: {response.text[:200]}")
                raise ValueError(f"REST API 연결 실패: {response.status_code}")
        except Exception as e_rest:
            print(f"⚠️  REST API 실패: {e_rest}")
            raise ValueError("SDK와 REST API 모두 실패했습니다.")


def init_gemini_sdk():
    """GEMINI SDK 방식 초기화"""
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 사용 가능한 모델 순서대로 시도 (최신 모델 우선)
    model_candidates = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    
    for candidate in model_candidates:
        try:
            model = genai.GenerativeModel(
                candidate,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'max_output_tokens': 2000,  # 토큰 제한 대폭 증가 (1000 -> 2000)
                }
            )
            test_response = model.generate_content("test")
            if test_response and test_response.text:
                print(f"✅ {candidate} 모델 사용 가능 (SDK)")
                return model
        except Exception as e:
            if '404' not in str(e) and '403' not in str(e):
                print(f"⚠️  {candidate} - {str(e)[:60]}")
            continue
    
    raise ValueError("사용 가능한 GEMINI 모델을 찾을 수 없습니다. API 키를 확인하세요.")


def generate_qa_pair_rest(chunk_text: str, api_config, max_retries: int = 3, i: int = 0) -> dict | None:
    """REST API를 사용한 QA 쌍 생성"""
    prompt = f"""### 지시
당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관] 내용을 바탕으로, 일반 고객이 실제로 물어볼 법한 자연스러운 질문과 그에 대한 답변을 생성하세요.

[요구사항]
1. 질문: 고객이 실제로 물어볼 법한 자연스러운 질문 (예: "이거 보상 되나요?", "침수됐는데 어떡하죠?")
2. 답변: 반드시 제공된 약관 내용에 기반하여 답변. 약관에 명시된 조항이나 근거를 언급하며 논리적으로 설명.
   - 약관에 없는 내용은 절대 지어내지 마세요.
   - 뉴스 기사, 광고, 외국어 등 무관한 내용을 포함하지 마세요.
   - 답변은 전문적이고 정확해야 합니다.
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용]
답변: [약관을 참조한 답변내용]

### 보험 약관 내용
{chunk_text[:1500]}

### 생성 결과
"""
    
    for attempt in range(max_retries):
        try:
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.9,
                    "topK": 40,
                    "maxOutputTokens": 2000,  # 1000 -> 2000으로 증가 (SDK와 동일하게)
                }
            }
            
            response = requests.post(api_config['api_url'], json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # 안전한 응답 파싱
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    
                    # finishReason 확인
                    finish_reason = candidate.get('finishReason', '')
                    if finish_reason == 'MAX_TOKENS':
                        print(f"⚠️  응답이 토큰 제한으로 잘렸습니다. maxOutputTokens를 늘려주세요.")
                    
                    # 표준 경로: candidates[0].content.parts[0].text
                    if 'content' in candidate and 'parts' in candidate['content']:
                        if len(candidate['content']['parts']) > 0:
                            part = candidate['content']['parts'][0]
                            if 'text' in part:
                                output = part['text']
                                return parse_qa_output(output, chunk_text)
                    
                    # 디버깅: 첫 번째 시도에서만 구조 출력
                    if attempt == 0 and i == 0:
                        print(f"🔍 응답 구조 디버깅:")
                        print(f"   candidate keys: {list(candidate.keys())}")
                        print(f"   finishReason: {finish_reason}")
                        if 'content' in candidate:
                            print(f"   content keys: {list(candidate['content'].keys())}")
                            if 'parts' not in candidate['content']:
                                print(f"   ⚠️  'parts' 키가 없습니다. 전체 응답:")
                                print(f"   {json.dumps(candidate, ensure_ascii=False, indent=2)[:800]}")
                    
                    # 대체 경로 1: candidates[0].text
                    if 'text' in candidate:
                        output = candidate['text']
                        return parse_qa_output(output, chunk_text)
                    
                    # 대체 경로 2: candidates[0].output 또는 candidates[0].content
                    if 'output' in candidate:
                        output = candidate['output']
                        return parse_qa_output(output, chunk_text)
                    
                    # 마지막 시도: content가 문자열인 경우
                    if 'content' in candidate and isinstance(candidate['content'], str):
                        output = candidate['content']
                        return parse_qa_output(output, chunk_text)
                
                # candidates가 없거나 비어있는 경우
                if attempt == 0:
                    print(f"⚠️  응답 구조가 예상과 다릅니다: {list(result.keys())}")
                    if 'candidates' in result:
                        print(f"   candidates 개수: {len(result.get('candidates', []))}")
            else:
                print(f"API 오류 (시도 {attempt + 1}): {response.status_code} - {response.text[:200]}")
                time.sleep(2)
        except KeyError as e:
            print(f"Attempt {attempt + 1} failed (KeyError): {e}")
            # 디버깅을 위해 응답 출력
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"   응답 구조: {list(result.keys())}")
                    if 'candidates' in result:
                        print(f"   candidates[0] 구조: {list(result['candidates'][0].keys()) if result['candidates'] else 'empty'}")
                except:
                    pass
            time.sleep(2)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    
    return None


def generate_qa_pair(chunk_text: str, model, max_retries: int = 3) -> dict | None:
    """
    약관 청크를 Context로 포함하여 RAG 스타일의 QA 쌍 생성 (GEMINI 사용)
    
    Returns:
        {"instruction": ..., "input": ..., "output": ...} 형태의 dict 또는 None
    """
    # RAG 스타일 프롬프트: 약관을 보고 답변하는 패턴 학습
    prompt = f"""### 지시
당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관] 내용을 바탕으로, 일반 고객이 실제로 물어볼 법한 자연스러운 질문과 그에 대한 답변을 생성하세요.

[요구사항]
1. 질문: 고객이 실제로 물어볼 법한 자연스러운 질문 (예: "이거 보상 되나요?", "침수됐는데 어떡하죠?")
2. 답변: 반드시 제공된 약관 내용에 기반하여 답변. 약관에 명시된 조항이나 근거를 언급하며 논리적으로 설명.
   - 약관에 없는 내용은 절대 지어내지 마세요.
   - 뉴스 기사, 광고, 외국어 등 무관한 내용을 포함하지 마세요.
   - 답변은 전문적이고 정확해야 합니다.
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용]
답변: [약관을 참조한 답변내용]

### 보험 약관 내용
{chunk_text[:1500]}

### 생성 결과
"""
    
    for attempt in range(max_retries):
        try:
            # GEMINI API 호출
            response = model.generate_content(prompt)
            
            # finish_reason 확인
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                
                if finish_reason == 2:  # MAX_TOKENS
                    print(f"⚠️  응답이 토큰 제한으로 잘렸습니다. max_output_tokens를 늘려주세요.")
                    # 일단 시도해보기
                    try:
                        output = response.text
                    except:
                        # parts에서 직접 추출 시도
                        if candidate.content and candidate.content.parts:
                            output = candidate.content.parts[0].text
                        else:
                            print(f"⚠️  응답이 비어있습니다. 재시도...")
                            time.sleep(2)
                            continue
                else:
                    output = response.text
            else:
                print(f"⚠️  응답에 candidates가 없습니다. 재시도...")
                time.sleep(2)
                continue
            
            # 파싱 함수 사용
            result = parse_qa_output(output, chunk_text)
            if result:
                return result
            else:
                print(f"⚠️  파싱 실패 또는 환각 감지, 재시도... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    
    return None


def parse_qa_output(output: str, chunk_text: str) -> dict | None:
    """생성된 출력에서 질문과 답변 파싱"""
    lines = output.strip().split('\n')
    question = None
    answer = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('질문:') or line.startswith('질문 :'):
            question = line.replace('질문:', '').replace('질문 :', '').strip()
        elif line.startswith('답변:') or line.startswith('답변 :'):
            answer = line.replace('답변:', '').replace('답변 :', '').strip()
            # 답변은 여러 줄일 수 있으므로 이후 줄들도 포함
            answer_lines = [answer]
            idx = lines.index(line) + 1
            while idx < len(lines) and not lines[idx].strip().startswith(('질문:', '답변:', '###')):
                if lines[idx].strip():
                    answer_lines.append(lines[idx].strip())
                idx += 1
            answer = '\n'.join(answer_lines)
            break
    
    if question and answer:
        # Validation: 환각 패턴 체크
        hallucination_patterns = [
            "이번에 새로 나온",
            "Este es",
            "Ich möchte",
            "평창동계올림픽",
            "마라톤",
            "원자력",
            "아니 형",
            "사랑해",
            "あ、もう"
        ]
        
        has_hallucination = any(pattern in answer for pattern in hallucination_patterns)
        
        if has_hallucination:
            return None  # 재시도 필요
        
        # RAG 스타일 Instruction Tuning 포맷
        return {
            "instruction": "아래 제공된 [보험약관]을 참고하여 사용자의 질문에 답변하세요. 약관에 명시된 내용에 근거하여 답변해야 하며, 약관에 없는 내용은 추측하지 말고 '약관에 해당 내용이 없습니다'라고 답변하세요.",
            "input": f"[보험약관]\n{chunk_text[:1500]}\n\n질문: {question}",
            "output": answer
        }
    
    return None


def load_chunks(jsonl_path: str, max_chunks: int = 600) -> list:
    """JSONL 파일에서 청크 로드 (랜덤 샘플링)"""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                chunk_text = item.get('text') or item.get('chunk') or item.get('content', '')
                if chunk_text and len(chunk_text) > 100:  # 너무 짧은 청크 제외
                    chunks.append(chunk_text)
            except json.JSONDecodeError:
                continue
    
    # 랜덤 샘플링
    if len(chunks) > max_chunks:
        chunks = random.sample(chunks, max_chunks)
    
    return chunks


def format_for_training(data: list) -> list:
    """
    Instruction Tuning 포맷으로 변환 (RAG 스타일)
    최종 형태: "### 지시\n{instruction}\n### 입력\n{input}\n### 출력\n{output}<|end_of_text|>"
    EOS 토큰을 명시적으로 추가하여 반복 생성 문제 해결
    """
    formatted = []
    for item in data:
        # EOS 토큰 명시적 추가
        text = f"### 지시\n{item['instruction']}\n### 입력\n{item['input']}\n### 출력\n{item['output']}<|end_of_text|>"
        formatted.append({"text": text})
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Generate RAG-style QA dataset using GEMINI 2.5 pro")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=600,
        help="Number of QA pairs to generate (default: 600)"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing file (skip already generated samples)"
    )
    args = parser.parse_args()

    # 기존 파일에서 이어서 생성 (resume 옵션)
    dataset = []
    start_idx = 0
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # 파일 형식 확인: {"text": "..."} 형식인지 {"instruction": ...} 형식인지
            if existing_data and isinstance(existing_data[0], dict):
                if 'text' in existing_data[0]:
                    # 이미 포맷팅된 형식이면 샘플 수만 카운트
                    start_idx = len(existing_data)
                    print(f"✅ 기존 파일에서 {start_idx}개 샘플 확인 (이미 포맷팅됨)")
                elif 'instruction' in existing_data[0]:
                    # 원본 형식이면 dataset에 추가
                    dataset = existing_data
                    start_idx = len(dataset)
                    print(f"✅ 기존 파일에서 {start_idx}개 샘플 로드 완료")
                else:
                    start_idx = len(existing_data)
                    print(f"✅ 기존 파일에서 {start_idx}개 샘플 확인")
            
            print(f"📊 {args.num_samples - start_idx}개 추가 생성 예정")
        except Exception as e:
            print(f"⚠️  기존 파일 로드 실패: {e}")
            print("새로 시작합니다...")
            dataset = []
            start_idx = 0

    # 1. 청크 로드
    print(f"Loading chunks from {DATA_PATH}...")
    chunks = load_chunks(DATA_PATH, max_chunks=args.num_samples * 2)  # 여유있게 로드
    print(f"Loaded {len(chunks)} chunks")

    # 2. GEMINI 모델 초기화
    model = init_gemini()
    print("✅ GEMINI 2.5 pro initialized")

    # 3. QA 쌍 생성 (RAG 스타일)
    remaining_samples = args.num_samples - start_idx
    if remaining_samples > 0:
        print(f"\nGenerating {remaining_samples} RAG-style QA pairs using GEMINI 2.5 pro...")
        if start_idx > 0:
            print(f"💡 이어서 생성: {start_idx}개 완료, {remaining_samples}개 남음")
        print("💡 개선 사항: GEMINI의 우수한 한국어 성능과 환각 감소 기능 활용")
    else:
        print(f"\n✅ 이미 {args.num_samples}개 샘플이 생성되어 있습니다!")
        return
    
    start_time = time.time()
    
    # 이어서 생성할 청크만 선택
    chunks_to_process = chunks[start_idx:start_idx + remaining_samples]
    
    for i, chunk in enumerate(tqdm(chunks_to_process, desc="Generating", initial=start_idx, total=args.num_samples)):
        # chunk가 dict인 경우 'text' 키에서 텍스트 추출
        chunk_text = chunk['text'] if isinstance(chunk, dict) else chunk
        
        # REST API 또는 SDK 방식에 따라 호출
        if isinstance(model, dict):
            qa_pair = generate_qa_pair_rest(chunk_text, model, i=i)
        else:
            qa_pair = generate_qa_pair(chunk_text, model)
        
        if qa_pair:
            dataset.append(qa_pair)
        
        # 중간 저장 (50개마다)
        if (start_idx + i + 1) % 50 == 0:
            # 기존 파일이 있으면 로드해서 병합
            if args.resume and os.path.exists(args.output):
                try:
                    with open(args.output, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    # 기존 데이터와 새 데이터 병합 (중복 제거)
                    all_data = existing_data + dataset
                    formatted_dataset = format_for_training(all_data)
                except:
                    formatted_dataset = format_for_training(dataset)
            else:
                formatted_dataset = format_for_training(dataset)
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_dataset, f, ensure_ascii=False, indent=2)
            print(f"\n💾 중간 저장 완료: {len(formatted_dataset)}개 저장됨")
        
        # API rate limit 방지 (요청 간 딜레이)
        time.sleep(0.5)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nGenerated {len(dataset)} valid QA pairs")
    print(f"⏱️  소요 시간: {elapsed_time/60:.1f}분")
    print(f"📊 성공률: {len(dataset)/remaining_samples*100:.1f}%")

    # 4. Instruction Tuning 포맷으로 변환 (EOS 토큰 포함)
    # 기존 파일과 병합
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            # 기존 데이터와 새 데이터 병합
            all_data = existing_data + dataset
            formatted_dataset = format_for_training(all_data)
            print(f"✅ 기존 {len(existing_data)}개 + 새로 생성 {len(dataset)}개 = 총 {len(formatted_dataset)}개")
        except Exception as e:
            print(f"⚠️  기존 파일 병합 실패: {e}")
            formatted_dataset = format_for_training(dataset)
    else:
        formatted_dataset = format_for_training(dataset)

    # 5. 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset saved to {output_path}")
    print(f"Total samples: {len(formatted_dataset)}")
    print("\n📊 샘플 데이터:")
    if formatted_dataset:
        print(json.dumps(formatted_dataset[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

