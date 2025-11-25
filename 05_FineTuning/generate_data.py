"""
파인튜닝용 QA 데이터셋 생성 스크립트
약관 청크를 기반으로 LLM에게 질문-답변 쌍을 생성하도록 요청
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# 설정
DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
OUTPUT_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset.json"


def load_4bit_model(model_name: str):
    """4-bit 양자화 모델 로드 (데이터 생성용)"""
    print(f"Loading data generation model: {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )
    
    return pipe


def generate_qa_pair(chunk_text: str, pipe, max_retries: int = 3) -> dict | None:
    """
    약관 청크를 기반으로 질문-답변 쌍 생성
    
    Returns:
        {"instruction": 질문, "input": 약관내용, "output": 답변} 형태의 dict 또는 None
    """
    prompt = f"""### 지시
아래 보험 약관 내용을 보고, 고객이 할 법한 '질문'과 그에 대한 '답변'을 1개만 만드세요.
형식은 반드시 다음과 같아야 합니다:
질문: [질문내용]
답변: [답변내용]

### 약관 내용
{chunk_text[:1000]}

### 생성 결과
"""
    
    for attempt in range(max_retries):
        try:
            output = pipe(prompt)[0]['generated_text']
            
            # 간단한 파싱 (질문과 답변 추출)
            lines = output.strip().split('\n')
            question = None
            answer = None
            
            for line in lines:
                if line.startswith('질문:') or line.startswith('질문 :'):
                    question = line.replace('질문:', '').replace('질문 :', '').strip()
                elif line.startswith('답변:') or line.startswith('답변 :'):
                    answer = line.replace('답변:', '').replace('답변 :', '').strip()
            
            if question and answer:
                return {
                    "instruction": question,
                    "input": chunk_text[:500],  # 약관 내용 일부
                    "output": answer
                }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    return None


def load_chunks(jsonl_path: str, max_chunks: int = 500) -> list:
    """JSONL 파일에서 청크 로드 (랜덤 샘플링)"""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chunks.append(item['text'])
    
    # 랜덤 샘플링
    if len(chunks) > max_chunks:
        chunks = random.sample(chunks, max_chunks)
    
    return chunks


def format_for_training(data: list) -> list:
    """
    Instruction Tuning 포맷으로 변환
    최종 형태: "### 지시\n{instruction}\n### 입력\n{input}\n### 출력\n{output}"
    """
    formatted = []
    for item in data:
        text = f"### 지시\n{item['instruction']}\n### 입력\n{item['input']}\n### 출력\n{item['output']}"
        formatted.append({"text": text})
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Generate QA dataset for fine-tuning")
    parser.add_argument(
        "--model",
        default="beomi/Llama-3-Open-Ko-8B",
        help="Model for data generation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of QA pairs to generate"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # 1. 청크 로드
    print(f"Loading chunks from {DATA_PATH}...")
    chunks = load_chunks(DATA_PATH, max_chunks=args.num_samples * 2)  # 여유있게 로드
    print(f"Loaded {len(chunks)} chunks")

    # 2. 모델 로드
    pipe = load_4bit_model(args.model)

    # 3. QA 쌍 생성
    print(f"\nGenerating {args.num_samples} QA pairs...")
    dataset = []
    
    for chunk in tqdm(chunks[:args.num_samples], desc="Generating"):
        qa_pair = generate_qa_pair(chunk, pipe)
        if qa_pair:
            dataset.append(qa_pair)
    
    print(f"Generated {len(dataset)} valid QA pairs")

    # 4. Instruction Tuning 포맷으로 변환
    formatted_dataset = format_for_training(dataset)

    # 5. 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {len(formatted_dataset)}")


if __name__ == "__main__":
    main()

