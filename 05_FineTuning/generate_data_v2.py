"""
파인튜닝용 QA 데이터셋 생성 스크립트 (RAG 스타일 - 개선 버전)
약관 청크를 Context로 포함하여 "문서를 보고 답변하는" 학습 데이터 생성
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
OUTPUT_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset_rag.json"


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
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )
    
    return pipe


def generate_qa_pair(chunk_text: str, pipe, max_retries: int = 3) -> dict | None:
    """
    약관 청크를 Context로 포함하여 RAG 스타일의 QA 쌍 생성
    
    Returns:
        {"instruction": ..., "input": ..., "output": ...} 형태의 dict 또는 None
    """
    # RAG 스타일 프롬프트: 약관을 보고 답변하는 패턴 학습
    prompt = f"""### 지시
당신은 보험 약관 전문가입니다. 아래 제공된 [보험 약관] 내용을 바탕으로, 일반 고객이 실제로 물어볼 법한 자연스러운 질문과 그에 대한 답변을 생성하세요.

[요구사항]
1. 질문: 고객이 실제로 물어볼 법한 자연스러운 질문 (예: "이거 보상 되나요?", "침수됐는데 어떡하죠?")
2. 답변: 반드시 제공된 약관 내용에 기반하여 답변. 약관에 명시된 조항이나 근거를 언급하며 논리적으로 설명.
3. 형식: 반드시 다음 형식으로 출력
질문: [질문내용]
답변: [약관을 참조한 답변내용]

### 보험 약관 내용
{chunk_text[:1500]}

### 생성 결과
"""
    
    for attempt in range(max_retries):
        try:
            output = pipe(prompt)[0]['generated_text']
            
            # 파싱 (질문과 답변 추출)
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
                # RAG 스타일 Instruction Tuning 포맷
                # instruction: 모델의 역할과 행동 지침
                # input: 참조할 약관 문서(Context) + 질문
                # output: 약관을 참조한 답변
                return {
                    "instruction": "아래 제공된 [보험약관]을 참고하여 사용자의 질문에 답변하세요. 약관에 명시된 내용에 근거하여 답변해야 하며, 약관에 없는 내용은 추측하지 말고 '약관에 해당 내용이 없습니다'라고 답변하세요.",
                    "input": f"[보험약관]\n{chunk_text[:1500]}\n\n질문: {question}",
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
            try:
                item = json.loads(line)
                # 'text' 필드가 있는 경우 그대로 사용, 없으면 다른 필드 확인
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
    parser = argparse.ArgumentParser(description="Generate RAG-style QA dataset for fine-tuning")
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

    # 3. QA 쌍 생성 (RAG 스타일)
    print(f"\nGenerating {args.num_samples} RAG-style QA pairs...")
    print("💡 개선 사항: 약관 Context를 포함하여 '문서를 보고 답변하는' 패턴 학습")
    dataset = []
    
    for i, chunk in enumerate(tqdm(chunks[:args.num_samples], desc="Generating")):
        qa_pair = generate_qa_pair(chunk, pipe)
        if qa_pair:
            dataset.append(qa_pair)
        
        # 중간 저장 (50개마다)
        if (i + 1) % 50 == 0:
            formatted_dataset = format_for_training(dataset)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_dataset, f, ensure_ascii=False, indent=2)
            print(f"\n💾 중간 저장 완료: {len(dataset)}개 생성됨")
    
    print(f"\nGenerated {len(dataset)} valid QA pairs")

    # 4. Instruction Tuning 포맷으로 변환 (EOS 토큰 포함)
    formatted_dataset = format_for_training(dataset)

    # 5. 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset saved to {output_path}")
    print(f"Total samples: {len(formatted_dataset)}")
    print("\n📊 샘플 데이터:")
    print(json.dumps(formatted_dataset[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

