"""
Base Model vs Fine-tuned Model (v1) vs Fine-tuned Model (v2) 평가 스크립트
재학습된 모델(v2)의 성능을 Base 모델과 v1 모델과 비교
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
LORA_PATH_V1 = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora"
LORA_PATH_V2 = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora-v2"
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


def load_finetuned_model(lora_path: str, model_name: str):
    """파인튜닝된 모델 로드 (Base + LoRA)"""
    print(f"Loading {model_name} model...")
    
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
    model = PeftModel.from_pretrained(base_model, lora_path)
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
    """모델 평가 실행 (최적화된 파라미터 적용)"""
    results = []
    
    # 환각 문구 감지용 stop sequences
    stop_phrases = ["이번에 새로 나온", "Este es", "Ich möchte", "이번엔", "이번에"]
    
    for i, query in enumerate(queries, 1):
        print(f"[{model_name}] Processing {i}/{len(queries)}: {query[:50]}...")
        
        # Retrieval
        docs = retriever.invoke(query)
        context = build_context(docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        
        # Generation (최적화된 파라미터 적용)
        output = pipe(
            prompt,
            repetition_penalty=1.2,  # 반복 생성 억제
            temperature=0.1,  # 사실 기반 답변 강제
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9
        )
        answer = output[0]['generated_text'].strip()
        
        # Stop sequences 후처리: 환각 문구가 포함된 경우 해당 부분 제거
        for phrase in stop_phrases:
            if phrase in answer:
                idx = answer.find(phrase)
                answer = answer[:idx].strip()
                break
        
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
    parser = argparse.ArgumentParser(description="Compare Base vs Fine-tuned v1 vs Fine-tuned v2 models")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of queries to evaluate (0 = all)"
    )
    parser.add_argument(
        "--output",
        default="/home/pencilfoxs/0_Insurance_PF/05_FineTuning/comparison_results_v2.json",
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
    
    # 메모리 정리
    del base_pipe
    torch.cuda.empty_cache()

    # 4. Fine-tuned 모델 (v1) 평가
    print("\n" + "="*60)
    print("Evaluating FINE-TUNED model (v1)...")
    print("="*60)
    
    if not Path(LORA_PATH_V1).exists():
        print(f"WARNING: LoRA model v1 not found at {LORA_PATH_V1}")
        print("Skipping v1 evaluation...")
        v1_results = []
    else:
        v1_pipe = load_finetuned_model(LORA_PATH_V1, "v1")
        v1_results = evaluate_model(v1_pipe, retriever, queries, "v1")
        del v1_pipe
        torch.cuda.empty_cache()

    # 5. Fine-tuned 모델 (v2) 평가
    print("\n" + "="*60)
    print("Evaluating FINE-TUNED model (v2 - RAG style)...")
    print("="*60)
    
    if not Path(LORA_PATH_V2).exists():
        print(f"ERROR: LoRA model v2 not found at {LORA_PATH_V2}")
        print("Please run train_qlora_v2.py first!")
        return
    
    v2_pipe = load_finetuned_model(LORA_PATH_V2, "v2")
    v2_results = evaluate_model(v2_pipe, retriever, queries, "v2")
    del v2_pipe
    torch.cuda.empty_cache()

    # 6. 결과 저장
    comparison = {
        "base_results": base_results,
        "finetuned_v1_results": v1_results,
        "finetuned_v2_results": v2_results,
        "queries": queries,
        "metadata": {
            "base_model": BASE_MODEL_ID,
            "lora_v1_path": LORA_PATH_V1,
            "lora_v2_path": LORA_PATH_V2,
            "num_queries": len(queries),
            "retriever": "BM25 (k=3)"
        }
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    print("\n=== Sample Comparison ===")
    print(f"Query: {queries[0]}")
    print(f"\nBase Answer:\n{base_results[0]['answer'][:200]}...")
    if v1_results:
        print(f"\nFine-tuned v1 Answer:\n{v1_results[0]['answer'][:200]}...")
    print(f"\nFine-tuned v2 Answer:\n{v2_results[0]['answer'][:200]}...")


if __name__ == "__main__":
    main()

