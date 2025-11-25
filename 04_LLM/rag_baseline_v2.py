"""
RAG Baseline Script v2 (4-bit Quantization 지원)
- Sparse Retriever (Mecab BM25)
- HuggingFace LLM with 4-bit quantization (default: beomi/Llama-3-Open-Ko-8B)
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "03_Retrieval"))

from bm25_retriever import MecabBM25Retriever  # noqa: E402


def load_retriever(k: int = 5) -> MecabBM25Retriever:
    """BM25 리트리버 로드"""
    retriever = MecabBM25Retriever.load_index(
        "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    )
    retriever.k = k
    return retriever


def load_4bit_model(model_name: str):
    """
    4-bit 양자화를 사용하여 모델 로드
    메모리 효율적이며 GPU 메모리가 부족한 환경에서도 대형 모델 실행 가능
    """
    print(f"Loading 4-bit Model: {model_name}...")
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Llama 모델의 경우 pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Pipeline 생성
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,  # 사실 기반 응답을 위해 낮춤
        top_p=0.9,
        return_full_text=False
    )
    
    print(f"Model loaded successfully on device: {next(model.parameters()).device}")
    return tokenizer, text_generation_pipeline


PROMPT_TEMPLATE = """### 지시
당신은 유능한 보험 전문가입니다. 아래 [참고 문서]를 바탕으로 사용자의 질문에 답변하세요.
문서에 없는 내용은 지어내지 말고 "정보가 없습니다"라고 말하세요.

### 참고 문서
{context}

### 질문
{question}

### 답변
"""


def build_context(docs) -> str:
    """검색된 문서를 프롬프트에 맞게 포맷팅"""
    chunks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"문서[{idx}] 회사: {meta.get('company', 'N/A')} | 경로: {meta.get('breadcrumbs', 'N/A')}"
        body = textwrap.shorten(doc.page_content, width=500, placeholder=" ...")
        chunks.append(f"{header}\n{body}")
    return "\n\n".join(chunks)


def generate_answer(pipeline, prompt: str) -> str:
    """LLM 파이프라인을 사용하여 답변 생성"""
    output = pipeline(prompt)
    return output[0]['generated_text'].strip()


def main():
    parser = argparse.ArgumentParser(description="RAG Baseline v2 (4-bit Quantization)")
    parser.add_argument(
        "--model",
        default="beomi/Llama-3-Open-Ko-8B",
        help="HuggingFace model name (recommended: beomi/Llama-3-Open-Ko-8B)"
    )
    parser.add_argument(
        "--question",
        default="음주운전하면 면책인가요?",
        help="User question in Korean"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of context documents"
    )
    args = parser.parse_args()

    # 1. Retriever 로드
    print("Loading BM25 retriever...")
    retriever = load_retriever(k=args.top_k)

    # 2. LLM 로드 (4-bit)
    tokenizer, pipe = load_4bit_model(args.model)

    # 3. RAG 실행
    print(f"\n===== Processing Query: {args.question} =====")
    docs = retriever.invoke(args.question)
    context = build_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=args.question)

    print("\n===== Prompt =====")
    print(prompt)
    print("==================\n")

    answer = generate_answer(pipe, prompt)
    print("===== Answer =====")
    print(answer)
    print("==================")


if __name__ == "__main__":
    main()

