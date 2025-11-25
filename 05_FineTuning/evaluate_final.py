"""
Base Model vs Fine-tuned Model 최종 평가 스크립트
동일한 질문 세트로 두 모델의 성능을 비교
"""

import argparse
import sys
import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "03_Retrieval"))

from bm25_retriever import MecabBM25Retriever  # noqa: E402


# 설정
BASE_MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
LORA_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora"
BM25_INDEX_PATH = "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
EVAL_DATASET_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/evaluation_dataset.json"


def load_retriever(k: int = 3):
    """BM25 리트리버 로드"""
    retriever = MecabBM25Retriever.load_index(BM25_INDEX_PATH)
    retriever.k = k
    return retriever


def load_base_model():
    """베이스 모델 로드 (4-bit)"""
    print("Loading BASE model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        return_full_text=False
    )
    
    return pipe


def load_finetuned_model():
    """파인튜닝된 모델 로드 (Base + LoRA)"""
    print("Loading FINE-TUNED model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()  # 어댑터를 베이스 모델에 병합
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        return_full_text=False
    )
    
    return pipe


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
    chunks = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"문서[{idx}] 회사: {meta.get('company', 'N/A')} | 경로: {meta.get('breadcrumbs', 'N/A')}"
        body = doc.page_content[:500]  # 처음 500자만
        chunks.append(f"{header}\n{body}")
    return "\n\n".join(chunks)


def evaluate_model(pipe, retriever, queries: List[str], model_name: str):
    """모델 평가 실행"""
    results = []
    
    for query in queries:
        # Retrieval
        docs = retriever.invoke(query)
        context = build_context(docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        
        # Generation
        output = pipe(prompt)
        answer = output[0]['generated_text'].strip()
        
        results.append({
            "query": query,
            "answer": answer,
            "context_sources": [doc.metadata.get('company', 'N/A') for doc in docs]
        })
    
    return results


def load_eval_queries() -> List[str]:
    """평가용 질문 로드"""
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['query'] for item in data]


def main():
    parser = argparse.ArgumentParser(description="Compare Base vs Fine-tuned models")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of queries to evaluate (0 = all)"
    )
    parser.add_argument(
        "--output",
        default="/home/pencilfoxs/0_Insurance_PF/05_FineTuning/comparison_results.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # 1. 평가 질문 로드
    queries = load_eval_queries()
    if args.limit > 0:
        queries = queries[:args.limit]
    print(f"Evaluating on {len(queries)} queries")

    # 2. Retriever 로드
    retriever = load_retriever(k=3)

    # 3. Base 모델 평가
    print("\n" + "="*60)
    print("Evaluating BASE model...")
    print("="*60)
    base_pipe = load_base_model()
    base_results = evaluate_model(base_pipe, retriever, queries, "base")

    # 4. Fine-tuned 모델 평가
    print("\n" + "="*60)
    print("Evaluating FINE-TUNED model...")
    print("="*60)
    
    if not Path(LORA_PATH).exists():
        print(f"ERROR: LoRA model not found at {LORA_PATH}")
        print("Please run train_qlora.py first!")
        return
    
    finetuned_pipe = load_finetuned_model()
    finetuned_results = evaluate_model(finetuned_pipe, retriever, queries, "finetuned")

    # 5. 결과 저장
    comparison = {
        "base_results": base_results,
        "finetuned_results": finetuned_results,
        "queries": queries
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\n=== Sample Comparison ===")
    print(f"Query: {queries[0]}")
    print(f"\nBase Answer:\n{base_results[0]['answer'][:200]}...")
    print(f"\nFine-tuned Answer:\n{finetuned_results[0]['answer'][:200]}...")


if __name__ == "__main__":
    main()

