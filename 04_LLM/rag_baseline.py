"""
RAG Baseline Script
 - Sparse Retriever (Mecab BM25)
 - HuggingFace LLM (default: google/gemma-2b-it)
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "03_Retrieval"))

from bm25_retriever import MecabBM25Retriever  # noqa: E402


def load_retriever(k: int = 5) -> MecabBM25Retriever:
    retriever = MecabBM25Retriever.load_index(
        "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    )
    retriever.k = k
    return retriever


def load_llm(model_name: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )
    return tokenizer, model


PROMPT_TEMPLATE = """당신은 국내 자동차 보험 전문가입니다.
다음 맥락(Context)을 참고하여 사용자의 질문에 정확하고 간결하게 답변하세요.
각 문단의 출처(보험사명, 문서 정보)를 명시하고, 정보가 없으면 추측하지 말고 '해당 약관 정보가 없습니다'라고 답하세요.

[Context]
{context}

[Question]
{question}

[Answer]
"""


def build_context(docs) -> str:
    chunks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[{idx}] 회사: {meta.get('company','N/A')} | 경로: {meta.get('breadcrumbs','N/A')}"
        body = textwrap.shorten(doc.page_content, width=500, placeholder=" ...")
        chunks.append(f"{header}\n{body}")
    return "\n\n".join(chunks)


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


def main():
    parser = argparse.ArgumentParser(description="RAG Baseline Runner")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model name")
    parser.add_argument("--question", required=True, help="User question in Korean")
    parser.add_argument("--top_k", type=int, default=3, help="Number of context documents")
    args = parser.parse_args()

    retriever = load_retriever(k=args.top_k)
    tokenizer, model = load_llm(args.model)

    docs = retriever.invoke(args.question)
    context = build_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=args.question)

    print("===== Prompt =====")
    print(prompt)
    print("==================\n")

    answer = generate_answer(model, tokenizer, prompt)
    print("===== Answer =====")
    print(answer)
    print("==================")


if __name__ == "__main__":
    main()

