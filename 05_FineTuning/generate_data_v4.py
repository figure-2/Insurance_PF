"""
v4 파인튜닝용 데이터셋 생성 (3,800개 - 확장 버전)
- 부정 사례: 760개 (20%)
- 긍정 사례: 1,900개 (50%)
- 복잡한 계산 시나리오: 570개 (15%)
- 조항 번호 명시: 380개 (10%)
- 동의어/검색 실패 대응: 190개 (5%)
- API: Google Gemini 2.0 Flash (속도 최적화)
"""

import json
import random
import argparse
import os
import time
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

# 설정
DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
OUTPUT_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset_v4_negative_enhanced.json"
GEMINI_MODEL = "gemini-2.0-flash"  # 변경: exp -> flash (속도 향상)
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# 목표 샘플 수
TARGET_NEGATIVE = 760
TARGET_POSITIVE = 1900
TARGET_CALCULATION = 570
TARGET_ARTICLE = 380
TARGET_SYNONYM = 190
TARGET_TOTAL = 3800


def load_api_key():
    """API 키 로드"""
    env_paths = ['/home/pencilfoxs/00_new/.env2', '/home/pencilfoxs/PJ/.env2']
    
    for path in env_paths:
        if os.path.exists(path):
            load_dotenv(path)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        for path in env_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        clean_line = line.split('#')[0].strip()
                        if clean_line.startswith("GOOGLE_API_KEY="):
                            api_key = clean_line.split('=', 1)[1].strip()
                            break
            if api_key:
                break
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env2 files.")
    
    return api_key


def classify_chunk(chunk_text: str) -> str:
    """청크를 부정/긍정/계산/조항으로 분류"""
    # 부정 키워드
    negative_keywords = [
        "보상하지 않", "제외", "면책", "불가", "불가능", 
        "거절", "해지", "무효", "보상하지 아니", "보상하지 않는"
    ]
    
    # 긍정 키워드
    positive_keywords = [
        "보상", "지급", "가능", "적용", "지원", "보장"
    ]
    
    # 계산 관련 키워드
    calculation_keywords = [
        "배율", "공제액", "한도", "지급보험금", "계산", "산출",
        "배분", "비율", "배수", "곱하기", "나누기", "합계"
    ]
    
    # 조항 번호 키워드
    article_keywords = [
        "제", "조", "항", "호", "목", "별표", "첨부"
    ]
    
    # 우선순위: 부정 > 계산 > 조항 > 긍정
    if any(kw in chunk_text for kw in negative_keywords) and (
        "보상하지 않" in chunk_text or "제외" in chunk_text or "면책" in chunk_text
    ):
        return "negative"
    
    if any(kw in chunk_text for kw in calculation_keywords):
        return "calculation"
    
    if any(kw in chunk_text for kw in article_keywords) and (
        "제" in chunk_text and ("조" in chunk_text or "항" in chunk_text)
    ):
        return "article"
    
    return "positive"


def call_gemini_api(api_key: str, payload: dict, max_retries: int = 5):
    """Gemini API 호출 (재시도 로직 포함)"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content
            elif response.status_code == 429:  # Rate limit
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"API Error {response.status_code}: {response.text}")
                time.sleep(2 ** attempt)
                continue
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    
    return None


def generate_negative_qa(chunk_text: str, api_key: str) -> dict | None:
    """부정 사례 QA 생성"""
    prompt = f"""당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관]에는 "보상하지 않는다", "제외한다", "면책" 등 **부정적인 내용**이 명시되어 있습니다.

**중요:** 약관에 "보상하지 않는다", "제외한다" 같은 부정 표현이 있으면, 반드시 그 내용을 명확히 언급하며 **"보상이 어렵습니다"**, **"보상받을 수 없습니다"** 같은 부정 표현을 사용하여 답변하세요.

[보험 약관]
{chunk_text[:2000]}

[요구사항]
1. 질문: 고객이 실제로 물어볼 법한 자연스러운 질문 (예: "이 경우 보상받을 수 있나요?")
2. 답변: 약관에 "보상하지 않는다", "제외한다" 등이 명시되어 있으면, 반드시 **"안타깝게도 보상이 어렵습니다"**, **"보상받을 수 없습니다"** 같은 부정 표현을 사용하세요.
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용]
답변: [약관을 참조한 답변내용 - 부정 표현 명확히 포함]

### 생성 결과
"""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 800
        }
    }
    
    output = call_gemini_api(api_key, payload)
    if not output:
        return None
    
    # 파싱
    lines = output.strip().split('\n')
    question = None
    answer = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('질문:') or line.startswith('질문 :'):
            question = line.replace('질문:', '').replace('질문 :', '').strip()
        elif line.startswith('답변:') or line.startswith('답변 :'):
            answer = line.replace('답변:', '').replace('답변 :', '').strip()
            # 답변은 여러 줄일 수 있음
            answer_lines = [answer]
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith(('질문:', '답변:', '###')):
                    answer_lines.append(next_line)
                else:
                    break
            answer = '\n'.join(answer_lines)
            break
    
    if question and answer:
        # 부정 표현 검증
        negative_phrases = ["보상이 어렵", "보상받을 수 없", "보상하지 않", "제외", "면책"]
        if any(phrase in answer for phrase in negative_phrases):
            return {
                "instruction": "아래 제공된 [보험약관]을 참고하여 사용자의 질문에 답변하세요. 약관에 명시된 내용에 근거하여 답변해야 하며, 약관에 없는 내용은 추측하지 말고 '약관에 해당 내용이 없습니다'라고 답변하세요.",
                "input": f"[보험약관]\n{chunk_text[:1500]}\n\n질문: {question}",
                "output": answer,
                "type": "negative"
            }
    
    return None


def generate_positive_qa(chunk_text: str, api_key: str) -> dict | None:
    """긍정 사례 QA 생성"""
    prompt = f"""당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관] 내용을 바탕으로, 일반 고객이 실제로 물어볼 법한 자연스러운 질문과 그에 대한 답변을 생성하세요.

[요구사항]
1. 질문: 고객이 실제로 물어볼 법한 자연스러운 질문 (예: "이거 보상 되나요?", "침수됐는데 어떡하죠?")
2. 답변: 반드시 제공된 약관 내용에 기반하여 답변. 약관에 명시된 조항이나 근거를 언급하며 논리적으로 설명.
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용]
답변: [약관을 참조한 답변내용]

[보험 약관]
{chunk_text[:2000]}

### 생성 결과
"""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 800
        }
    }
    
    output = call_gemini_api(api_key, payload)
    if not output:
        return None
    
    # 파싱
    lines = output.strip().split('\n')
    question = None
    answer = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('질문:') or line.startswith('질문 :'):
            question = line.replace('질문:', '').replace('질문 :', '').strip()
        elif line.startswith('답변:') or line.startswith('답변 :'):
            answer = line.replace('답변:', '').replace('답변 :', '').strip()
            answer_lines = [answer]
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith(('질문:', '답변:', '###')):
                    answer_lines.append(next_line)
                else:
                    break
            answer = '\n'.join(answer_lines)
            break
    
    if question and answer:
        return {
            "instruction": "아래 제공된 [보험약관]을 참고하여 사용자의 질문에 답변하세요. 약관에 명시된 내용에 근거하여 답변해야 하며, 약관에 없는 내용은 추측하지 말고 '약관에 해당 내용이 없습니다'라고 답변하세요.",
            "input": f"[보험약관]\n{chunk_text[:1500]}\n\n질문: {question}",
            "output": answer,
            "type": "positive"
        }
    
    return None


def generate_calculation_qa(chunk_text: str, api_key: str) -> dict | None:
    """복잡한 계산 시나리오 QA 생성"""
    prompt = f"""당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관]에는 보험금 계산 공식, 배율, 공제액, 한도 등 **계산이 필요한 내용**이 포함되어 있습니다.

**중요:** 약관에 명시된 계산 공식(예: 지급보험금 = 실제손해액 + 비용 - 공제액)을 사용하여 구체적인 시나리오의 보험금을 계산하는 질문과 답변을 생성하세요.

[보험 약관]
{chunk_text[:2000]}

[요구사항]
1. 질문: 구체적인 수치가 포함된 시나리오 질문 (예: "보험가입금액 3천만원, 후유장애 7급일 때 보험금은?")
2. 답변: 약관의 계산 공식을 단계별로 설명하고, 최종 금액을 명확히 제시하세요.
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용]
답변: [단계별 계산 과정 + 최종 금액]

### 생성 결과
"""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 1000
        }
    }
    
    output = call_gemini_api(api_key, payload)
    if not output:
        return None
    
    # 파싱
    lines = output.strip().split('\n')
    question = None
    answer = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('질문:') or line.startswith('질문 :'):
            question = line.replace('질문:', '').replace('질문 :', '').strip()
        elif line.startswith('답변:') or line.startswith('답변 :'):
            answer = line.replace('답변:', '').replace('답변 :', '').strip()
            answer_lines = [answer]
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith(('질문:', '답변:', '###')):
                    answer_lines.append(next_line)
                else:
                    break
            answer = '\n'.join(answer_lines)
            break
    
    if question and answer:
        # 계산 관련 키워드 검증
        calculation_keywords = ["만원", "원", "계산", "배율", "공제", "한도", "합계", "곱하기", "나누기"]
        if any(kw in answer for kw in calculation_keywords):
            return {
                "instruction": "아래 제공된 [보험약관]을 참고하여 사용자의 질문에 답변하세요. 약관에 명시된 내용에 근거하여 답변해야 하며, 약관에 없는 내용은 추측하지 말고 '약관에 해당 내용이 없습니다'라고 답변하세요.",
                "input": f"[보험약관]\n{chunk_text[:1500]}\n\n질문: {question}",
                "output": answer,
                "type": "calculation"
            }
    
    return None


def generate_article_qa(chunk_text: str, api_key: str) -> dict | None:
    """조항 번호 명시 QA 생성"""
    prompt = f"""당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관]에는 "제X조", "제X항" 같은 **조항 번호**가 명시되어 있습니다.

**중요:** 답변 시 반드시 약관의 조항 번호를 명시하세요 (예: "약관 제16조 4항에 따르면...").

[보험 약관]
{chunk_text[:2000]}

[요구사항]
1. 질문: 조항 번호를 물어보는 질문 또는 조항 번호를 언급해야 하는 질문
2. 답변: 반드시 "약관 제X조", "제X항" 같은 조항 번호를 명시하여 답변하세요.
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용]
답변: [조항 번호를 명시한 답변내용]

### 생성 결과
"""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 800
        }
    }
    
    output = call_gemini_api(api_key, payload)
    if not output:
        return None
    
    # 파싱
    lines = output.strip().split('\n')
    question = None
    answer = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('질문:') or line.startswith('질문 :'):
            question = line.replace('질문:', '').replace('질문 :', '').strip()
        elif line.startswith('답변:') or line.startswith('답변 :'):
            answer = line.replace('답변:', '').replace('답변 :', '').strip()
            answer_lines = [answer]
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith(('질문:', '답변:', '###')):
                    answer_lines.append(next_line)
                else:
                    break
            answer = '\n'.join(answer_lines)
            break
    
    if question and answer:
        # 조항 번호 검증
        if "제" in answer and ("조" in answer or "항" in answer):
            return {
                "instruction": "아래 제공된 [보험약관]을 참고하여 사용자의 질문에 답변하세요. 약관에 명시된 내용에 근거하여 답변해야 하며, 약관에 없는 내용은 추측하지 말고 '약관에 해당 내용이 없습니다'라고 답변하세요.",
                "input": f"[보험약관]\n{chunk_text[:1500]}\n\n질문: {question}",
                "output": answer,
                "type": "article"
            }
    
    return None


def generate_synonym_qa(chunk_text: str, api_key: str, synonym_dict: dict = None) -> dict | None:
    """동의어/검색 실패 대응 QA 생성"""
    # 동의어 사전 (확장 버전 - 보험 약관에서 자주 나오는 단어 중심)
    if synonym_dict is None:
        synonym_dict = {
            # 기존 키워드
            "노트북": ["휴대용 컴퓨터", "랩톱", "노트북 PC", "노트북 컴퓨터"],
            "아내": ["배우자", "부인", "처", "와이프"],
            "전업주부": ["가사종사자", "무직", "주부", "가정주부"],
            "휴대품": ["소지품", "짐", "물건", "가방"],
            "파손": ["손상", "손해", "깨짐", "망가짐"],
            
            # [확장] 보험 약관에서 자주 나오는 핵심 단어들
            "자동차": ["차", "차량", "제 차", "내 차", "자가용", "승용차"],
            "사고": ["부딪힘", "충돌", "박음", "접촉사고", "꽝", "사건"],
            "피보험자": ["가입자", "보험 든 사람", "계약자", "보험 가입한 사람"],
            "보험자": ["보험회사", "회사", "보험사", "보험 업체"],
            "약관": ["계약서", "설명서", "규정집", "보험 책자", "조건"],
            "보험금": ["보상금", "돈", "치료비", "수리비", "지급금"],
            "보상": ["배상", "지급", "돈 받기", "보상받기"],
            "지급": ["받기", "지급받기", "돈 받기", "지급해주기"],
            "손해": ["피해", "망가짐", "손상", "부상"],
            "계약": ["가입", "든 거", "약정", "계약하기"],
            "가입": ["들기", "계약", "신청", "보험 들기"],
            "기간": ["언제까지", "날짜", "유효기간", "기한"],
            "해지": ["취소", "그만둠", "탈퇴", "환불", "해약"],
            "운전": ["드라이브", "주행", "몰다", "운전하다"],
            "면허": ["라이선스", "운전증", "자격증", "면허증"],
            "청구": ["신고", "접수", "요청", "신청"],
            "수리": ["고치기", "수리받기", "보수", "수선"],
            "교체": ["바꾸기", "교환", "새것으로 바꾸기"],
            "도난": ["훔쳐감", "도둑", "없어짐", "도둑맞음"],
            "침수": ["물에 잠김", "홍수", "물바다", "침수됨"],
            "화재": ["불", "불남", "전소", "불타버림"],
            "한도": ["최대 금액", "한계", "상한선", "최대 한도"],
            "공제": ["빼기", "차감", "공제액", "빼는 금액"],
            "자녀": ["아들", "딸", "애기", "자식", "자식들"],
            "가족": ["식구", "애들", "부모님", "가족들"],
            "의사": ["선생님", "주치의", "전문의", "의료진"],
            "병원": ["응급실", "의원", "대학병원", "치료받은 곳"],
            "부품": ["파츠", "부속", "부분", "기계 부품"],
            "타이어": ["바퀴", "휠", "타이어"],
            "배터리": ["밧데리", "충전지", "전지"]
        }
    
    # 청크에서 키워드 찾기
    found_keywords = []
    for key, synonyms in synonym_dict.items():
        if key in chunk_text:
            found_keywords.append((key, synonyms))
    
    if not found_keywords:
        # 키워드가 없으면 일반 긍정 QA 생성
        return generate_positive_qa(chunk_text, api_key)
    
    # 첫 번째 발견된 키워드의 동의어로 질문 변형
    keyword, synonyms = found_keywords[0]
    synonym = random.choice(synonyms)
    
    prompt = f"""당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관] 내용을 바탕으로, 일반 고객이 실제로 물어볼 법한 자연스러운 질문과 그에 대한 답변을 생성하세요.

**중요:** 질문에서 "{keyword}" 대신 "{synonym}" 같은 동의어나 유사 표현을 사용하세요. 하지만 답변은 약관의 원래 용어를 사용하세요.

[보험 약관]
{chunk_text[:2000]}

[요구사항]
1. 질문: "{synonym}" 같은 동의어를 사용한 자연스러운 질문
2. 답변: 약관의 원래 용어("{keyword}")를 사용하여 정확히 답변
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용 - 동의어 사용]
답변: [약관을 참조한 답변내용 - 원래 용어 사용]

### 생성 결과
"""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 800
        }
    }
    
    output = call_gemini_api(api_key, payload)
    if not output:
        return None
    
    # 파싱
    lines = output.strip().split('\n')
    question = None
    answer = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('질문:') or line.startswith('질문 :'):
            question = line.replace('질문:', '').replace('질문 :', '').strip()
        elif line.startswith('답변:') or line.startswith('답변 :'):
            answer = line.replace('답변:', '').replace('답변 :', '').strip()
            answer_lines = [answer]
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith(('질문:', '답변:', '###')):
                    answer_lines.append(next_line)
                else:
                    break
            answer = '\n'.join(answer_lines)
            break
    
    if question and answer:
        return {
            "instruction": "아래 제공된 [보험약관]을 참고하여 사용자의 질문에 답변하세요. 약관에 명시된 내용에 근거하여 답변해야 하며, 약관에 없는 내용은 추측하지 말고 '약관에 해당 내용이 없습니다'라고 답변하세요.",
            "input": f"[보험약관]\n{chunk_text[:1500]}\n\n질문: {question}",
            "output": answer,
            "type": "synonym"
        }
    
    return None


def load_and_classify_chunks(jsonl_path: str) -> dict:
    """청크를 로드하고 유형별로 분류"""
    chunks = {
        "negative": [],
        "positive": [],
        "calculation": [],
        "article": []
    }
    
    print("Loading and classifying chunks...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                chunk_text = item.get('text') or item.get('chunk') or item.get('content', '')
                if chunk_text and len(chunk_text) > 100:
                    chunk_type = classify_chunk(chunk_text)
                    if chunk_type in chunks:
                        chunks[chunk_type].append(chunk_text)
            except json.JSONDecodeError:
                continue
    
    print(f"Classified chunks:")
    print(f"  - Negative: {len(chunks['negative'])}")
    print(f"  - Positive: {len(chunks['positive'])}")
    print(f"  - Calculation: {len(chunks['calculation'])}")
    print(f"  - Article: {len(chunks['article'])}")
    
    return chunks

def load_existing_data(file_path: str) -> list:
    """기존 데이터 로드"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ 기존 데이터 {len(data)}개를 로드했습니다.")
                return data
        except json.JSONDecodeError:
            print("⚠️ 기존 파일이 손상되었거나 비어있습니다. 새로 시작합니다.")
            return []
    return []

def main():
    parser = argparse.ArgumentParser(description="Generate v4 dataset with 3,800 samples")
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--only-synonym",
        action="store_true",
        help="Generate only synonym data (skip other types)"
    )
    args = parser.parse_args()

    # 1. API 키 로드
    print("Loading API key...")
    api_key = load_api_key()
    print("✅ API Key loaded")
    print(f"📌 Using model: {GEMINI_MODEL}")

    # 2. 청크 로드 및 분류
    chunks = load_and_classify_chunks(DATA_PATH)
    
    # 3. 기존 데이터 로드 및 진행 상황 파악
    dataset = load_existing_data(args.output)
    stats = defaultdict(int)
    for item in dataset:
        stats[item.get('type', 'unknown')] += 1
    
    print(f"\n📊 Current Status (Loaded):")
    for k, v in stats.items():
        print(f"   - {k}: {v}")
    
    # 4. 남은 수량 계산
    if args.only_synonym:
        # Synonym만 재생성
        remaining_targets = {
            "negative": 0,
            "positive": 0,
            "calculation": 0,
            "article": 0,
            "synonym": max(0, TARGET_SYNONYM - stats["synonym"]),
        }
        print(f"\n🔄 Synonym만 재생성 모드: {remaining_targets['synonym']}개 필요")
    else:
        remaining_targets = {
            "negative": max(0, TARGET_NEGATIVE - stats["negative"]),
            "positive": max(0, TARGET_POSITIVE - stats["positive"]),
            "calculation": max(0, TARGET_CALCULATION - stats["calculation"]),
            "article": max(0, TARGET_ARTICLE - stats["article"]),
            "synonym": max(0, TARGET_SYNONYM - stats["synonym"]),
        }

    # 5. 샘플링 (부족한 만큼만)
    sampled_chunks = {}
    
    # Negative
    if remaining_targets["negative"] > 0:
        available = chunks["negative"]
        count = min(remaining_targets["negative"], len(available))
        sampled_chunks["negative"] = random.sample(available, count)
    else:
        sampled_chunks["negative"] = []

    # Positive
    if remaining_targets["positive"] > 0:
        available = chunks["positive"]
        count = min(remaining_targets["positive"], len(available))
        sampled_chunks["positive"] = random.sample(available, count)
    else:
        sampled_chunks["positive"] = []

    # Calculation
    if remaining_targets["calculation"] > 0:
        available = chunks["calculation"]
        count = min(remaining_targets["calculation"], len(available))
        sampled_chunks["calculation"] = random.sample(available, count)
    else:
        sampled_chunks["calculation"] = []

    # Article
    if remaining_targets["article"] > 0:
        available = chunks["article"]
        count = min(remaining_targets["article"], len(available))
        sampled_chunks["article"] = random.sample(available, count)
    else:
        sampled_chunks["article"] = []
    
    # Synonym (Positive에서 샘플링)
    if remaining_targets["synonym"] > 0:
        available = chunks["positive"]
        count = min(remaining_targets["synonym"], len(available))
        sampled_chunks["synonym"] = random.sample(available, count)
    else:
        sampled_chunks["synonym"] = []

    
    print(f"\n🚀 Generation Plan (Remaining):")
    print(f"   - Negative: {len(sampled_chunks['negative'])} chunks")
    print(f"   - Positive: {len(sampled_chunks['positive'])} chunks")
    print(f"   - Calculation: {len(sampled_chunks['calculation'])} chunks")
    print(f"   - Article: {len(sampled_chunks['article'])} chunks")
    print(f"   - Synonym: {len(sampled_chunks['synonym'])} chunks")
    
    # 6. 추가 데이터 생성
    
    # 6-1. 부정 사례 생성 (--only-synonym 모드에서는 스킵)
    if not args.only_synonym and sampled_chunks['negative']:
        print(f"\n🔴 Generating {len(sampled_chunks['negative'])} negative QA pairs...")
        for chunk in tqdm(sampled_chunks['negative'], desc="Negative"):
            qa_pair = generate_negative_qa(chunk, api_key)
            if qa_pair:
                dataset.append(qa_pair)
                stats['negative'] += 1
            
            if len(dataset) % 50 == 0:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"💾 중간 저장: {len(dataset)}개 (Negative 완료: {stats['negative']})")
            time.sleep(0.1) # 속도 빨라졌으므로 딜레이 감소

    # 6-2. 긍정 사례 생성 (--only-synonym 모드에서는 스킵)
    if not args.only_synonym and sampled_chunks['positive']:
        print(f"\n🟢 Generating {len(sampled_chunks['positive'])} positive QA pairs...")
        for chunk in tqdm(sampled_chunks['positive'], desc="Positive"):
            qa_pair = generate_positive_qa(chunk, api_key)
            if qa_pair:
                dataset.append(qa_pair)
                stats['positive'] += 1
            
            if len(dataset) % 50 == 0:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"💾 중간 저장: {len(dataset)}개 (Positive 완료: {stats['positive']})")
            time.sleep(0.1)

    # 6-3. 계산 시나리오 생성 (--only-synonym 모드에서는 스킵)
    if not args.only_synonym and sampled_chunks['calculation']:
        print(f"\n🔢 Generating {len(sampled_chunks['calculation'])} calculation QA pairs...")
        for chunk in tqdm(sampled_chunks['calculation'], desc="Calculation"):
            qa_pair = generate_calculation_qa(chunk, api_key)
            if qa_pair:
                dataset.append(qa_pair)
                stats['calculation'] += 1
            
            if len(dataset) % 50 == 0:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"💾 중간 저장: {len(dataset)}개 (Calculation 완료: {stats['calculation']})")
            time.sleep(0.1)

    # 6-4. 조항 번호 명시 생성 (--only-synonym 모드에서는 스킵)
    if not args.only_synonym and sampled_chunks['article']:
        print(f"\n📋 Generating {len(sampled_chunks['article'])} article QA pairs...")
        for chunk in tqdm(sampled_chunks['article'], desc="Article"):
            qa_pair = generate_article_qa(chunk, api_key)
            if qa_pair:
                dataset.append(qa_pair)
                stats['article'] += 1
            
            if len(dataset) % 50 == 0:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"💾 중간 저장: {len(dataset)}개 (Article 완료: {stats['article']})")
            time.sleep(0.1)

    # 6-5. 동의어/검색 실패 대응 생성
    if sampled_chunks['synonym']:
        print(f"\n🔄 Generating {len(sampled_chunks['synonym'])} synonym QA pairs...")
        print(f"⏳ 천천히 진행 중... (API 호출 간격: 0.5초)")
        for chunk in tqdm(sampled_chunks['synonym'], desc="Synonym"):
            qa_pair = generate_synonym_qa(chunk, api_key)
            if qa_pair:
                dataset.append(qa_pair)
                stats['synonym'] += 1
            else:
                print(f"⚠️ Synonym 생성 실패 (청크 스킵)")
            
            # 중간 저장 (더 자주 저장)
            if len(dataset) % 25 == 0:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"💾 중간 저장: {len(dataset)}개 (Synonym 완료: {stats['synonym']}/{len(sampled_chunks['synonym'])})")
            time.sleep(0.5)  # 천천히 진행 (0.1초 -> 0.5초)

    # 7. 최종 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 8. 통계 출력
    print(f"\n✅ Dataset saved to {args.output}")
    print(f"\n📊 Final Statistics:")
    print(f"   - Total: {len(dataset)} QA pairs")
    for type_name, count in stats.items():
        print(f"   - {type_name.capitalize()}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # 샘플 출력
    print(f"\n📝 Sample (Negative):")
    negative_samples = [item for item in dataset if item.get('type') == 'negative']
    if negative_samples:
        print(json.dumps(negative_samples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


