"""
LLM Baseline Evaluation Runner
 - 여러 질문을 RAG 파이프라인으로 실행하고 결과를 CSV로 저장
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "03_Retrieval"))

from bm25_retriever import MecabBM25Retriever  # noqa: E402
from rag_baseline import load_llm, build_context, PROMPT_TEMPLATE, generate_answer  # noqa: E402

EVAL_DATASET = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/evaluation_dataset.json"


def load_dataset(path: str, limit: int | None = None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def evaluate_model(model_name: str, questions: List[Dict], top_k: int = 3, output_csv: str = "llm_baseline_results.csv"):
    retriever = MecabBM25Retriever.load_index(
        "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    )
    retriever.k = top_k

    tokenizer, model = load_llm(model_name)

    rows = []
    for idx, sample in enumerate(questions, start=1):
        query = sample["query"]

        start = time.time()
        docs = retriever.invoke(query)
        context = build_context(docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        answer = generate_answer(model, tokenizer, prompt)
        duration = time.time() - start

        rows.append({
            "id": idx,
            "query": query,
            "answer": answer,
            "context_sources": " | ".join(doc.metadata.get("company", "N/A") for doc in docs[:top_k]),
            "time": f"{duration:.2f}",
        })
        print(f"[{idx}/{len(questions)}] {query} -> {duration:.2f}s")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="LLM Baseline Evaluator")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--limit", type=int, default=10, help="Number of queries to evaluate")
    parser.add_argument("--output", default="llm_baseline_results.csv")
    args = parser.parse_args()

    dataset = load_dataset(EVAL_DATASET, limit=args.limit)
    evaluate_model(args.model, dataset, output_csv=args.output)


if __name__ == "__main__":
    main()

