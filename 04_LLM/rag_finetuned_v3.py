"""
RAG with Fine-Tuned Model v3
- Retriever: Mecab BM25
- Generator: Llama-3-Ko-8B + LoRA (v3) (4-bit Quantization)
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "03_Retrieval"))

from bm25_retriever import MecabBM25Retriever  # noqa: E402


def load_retriever(k: int = 3) -> MecabBM25Retriever:
    """BM25 리트리버 로드"""
    print(f"Loading BM25 retriever (top_k={k})...")
    retriever = MecabBM25Retriever.load_index(
        "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    )
    retriever.k = k
    return retriever


def load_finetuned_model():
    """Fine-Tuned Model (v3) 로드"""
    base_model_id = "beomi/Llama-3-Open-Ko-8B"
    adapter_path = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora-v3"
    
    print(f"Loading Base Model: {base_model_id} (4-bit)...")
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Base Model 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # LoRA 어댑터 로드
    print(f"Loading LoRA Adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def build_prompt(context: str, question: str) -> str:
    """RAG 프롬프트 구성 (학습 데이터 포맷과 일치)"""
    return f"""### 지시
{question}

### 입력
{context}

### 출력
"""


def generate_answer(model, tokenizer, prompt: str) -> str:
    """답변 생성"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,  # 사실 기반 응답을 위해 낮춤
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 부분 제거하고 답변만 추출
    if "### 출력" in generated_text:
        answer = generated_text.split("### 출력")[1].strip()
    elif "### 답변" in generated_text:
        answer = generated_text.split("### 답변")[1].strip()
    else:
        # 프롬프트 길이만큼 잘라내기 (fallback)
        prompt_len = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        answer = generated_text[prompt_len:].strip()
        
    return answer


def format_context(docs) -> str:
    """검색된 문서 포맷팅"""
    context_parts = []
    for idx, doc in enumerate(docs, 1):
        content = doc.page_content
        context_parts.append(content)
    return "\n\n".join(context_parts)


def main():
    parser = argparse.ArgumentParser(description="RAG with Fine-Tuned Model v3")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved documents")
    args = parser.parse_args()

    # 1. 리트리버 로드
    retriever = load_retriever(k=args.top_k)

    # 2. 모델 로드
    model, tokenizer = load_finetuned_model()

    # 3. 검색 실행
    print(f"\nSearching... (Question: {args.question})")
    docs = retriever.invoke(args.question)
    context = format_context(docs)

    # 4. 프롬프트 생성
    prompt = build_prompt(context, args.question)
    
    # 5. 답변 생성
    print("Generating answer...")
    answer = generate_answer(model, tokenizer, prompt)
    
    print("\n" + "="*50)
    print(f"Question: {args.question}")
    print("-" * 50)
    print(f"Retrieved Context ({len(docs)} docs):")
    for idx, doc in enumerate(docs, 1):
        print(f"[{idx}] {doc.page_content[:200]}...")
    print("-" * 50)
    print(f"Answer:\n{answer}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

