"""
임베딩 모델 후보 비교 스크립트
 - 기존 30개 FAQ + 추가 20개(동의어/복합/절차) = 총 50개 평가
 - 지표: Avg Positive Score, Avg Separation(양수 권장), Recall(Separation > 0 비율), Time/Item
"""

import json
import os
import time
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

BASE_DATASET_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/evaluation_dataset.json"

ADDITIONAL_QUERIES: List[Dict[str, str]] = [
    {
        "query": "뺑소니 사고 보상되나요?",
        "positive": "보유불명자동차에 의한 사고는 자기차량손해 담보로 보상받을 수 있으나 할증이 적용됩니다.",
        "negative": "보험료 할인은 Eco 마일리지 특약 가입 시에만 가능합니다."
    },
    {
        "query": "자차부담금 얼마예요?",
        "positive": "자기차량손해 담보에서는 보험증권에 기재된 최소·최대 자기부담금 한도로 손해액을 부담합니다.",
        "negative": "대인배상Ⅰ은 운전자 연령에 관계없이 의무적으로 보상합니다."
    },
    {
        "query": "대리기사 사고 보상",
        "positive": "대리운전자가 운전하다 사고를 내더라도 대인배상Ⅱ와 대물배상(의무)은 보상 가능합니다.",
        "negative": "Eco 마일리지 특약은 주행거리가 짧은 경우 보험료를 할인해 주는 제도입니다."
    },
    {
        "query": "견인 거리 얼마나 되나요?",
        "positive": "긴급견인 서비스는 10km까지 무료이며, 초과 거리에는 요금이 발생합니다.",
        "negative": "물적사고 할증기준은 50만원 이상 지급 시 적용됩니다."
    },
    {
        "query": "배터리 방전됐어요",
        "positive": "긴급출동 서비스로 배터리 충전 및 점프 서비스를 무료로 제공합니다.",
        "negative": "자동차 보험증권은 인터넷에서도 발급받을 수 있습니다."
    },
    {
        "query": "여행 중 렌터카 빌렸는데 내 보험 되나요?",
        "positive": "다른자동차운전담보특약에 가입했다면 렌터카 운전 중 발생한 손해도 보상 가능합니다.",
        "negative": "기명피보험자 1인 한정 특약은 본인만 운전할 수 있도록 제한합니다."
    },
    {
        "query": "가족이 내 차 몰다가 사고나면?",
        "positive": "운전자 범위가 가족 한정이라면 가족이 운전 중 발생한 사고도 보상합니다.",
        "negative": "대물배상은 상대 차량 수리비만 보상하고 기타 비용은 보상하지 않습니다."
    },
    {
        "query": "차유리 돌 맞아서 깨졌는데 보상되나요?",
        "positive": "자기차량손해 담보에서는 비산물로 인한 앞유리 손상도 보상합니다.",
        "negative": "보험 기간 중 주소가 변경되면 15일 이내에 통지해야 합니다."
    },
    {
        "query": "태풍으로 침수되면 보상되나요?",
        "positive": "자연재해(태풍·홍수)로 인한 침수 피해는 자기차량손해 담보에서 보상합니다.",
        "negative": "무면허운전 사고는 약관상 면책 사항으로 보상하지 않습니다."
    },
    {
        "query": "문콕 당했는데 상대방을 못 찾으면?",
        "positive": "가해차량을 특정할 수 없는 물적사고의 경우 자기차량손해로 처리하며 할증이 적용될 수 있습니다.",
        "negative": "운전자 연령 한정 특약은 26세 이상 운전자만 운전할 수 있도록 제한합니다."
    },
    {
        "query": "음주운전 부담금 얼마?",
        "positive": "음주운전 사고 시 대인배상Ⅱ는 사고부담금 1억원, 대물배상은 5천만원을 부담해야 합니다.",
        "negative": "긴급출동 서비스는 연간 5회까지만 이용할 수 있습니다."
    },
    {
        "query": "무면허운전 부담금",
        "positive": "무면허 운전 사고 시에도 대인배상Ⅱ 사고부담금 1억원을 부담해야 합니다.",
        "negative": "Eco 마일리지 특약은 연간 주행거리 2만km 이하인 경우 할인됩니다."
    },
    {
        "query": "할증 기준 금액이 얼마인가요?",
        "positive": "물적사고 할증은 보험금 지급액이 50만원 이상일 때 적용됩니다.",
        "negative": "자기신체사고 담보는 치료비만 보상하고 위자료는 보상하지 않습니다."
    },
    {
        "query": "긴급출동 몇 번 부를 수 있나요?",
        "positive": "긴급출동 서비스는 연간 5회까지 무료로 제공됩니다.",
        "negative": "임시운전자특약은 특정 기간 동안 운전자 범위를 일시적으로 넓히는 제도입니다."
    },
    {
        "query": "대물배상 최소 가입금액",
        "positive": "대물배상의 의무보험 가입금액은 자동차손해배상보장법 시행령에 따라 2천만원입니다.",
        "negative": "보험 계약 해지는 약관상 정해진 사유가 있을 때만 가능합니다."
    },
    {
        "query": "보험금 청구 서류 뭐 필요해?",
        "positive": "보험금 청구 시 청구서, 손해입증서류, 신분증 사본 등이 필요합니다.",
        "negative": "차량가액은 보험개발원이 정한 차량기준가액표를 따릅니다."
    },
    {
        "query": "가지급금 받을 수 있나요?",
        "positive": "보험금 확정 전이라도 손해액이 뚜렷하면 가지급금을 지급받을 수 있습니다.",
        "negative": "주소 변경 시 15일 내 통지하지 않으면 계약 해지가 될 수 있습니다."
    },
    {
        "query": "보험료 분할 납부 되나요?",
        "positive": "보험료는 약관에서 정한 방법에 따라 분할 납입이 가능합니다.",
        "negative": "무보험자동차상해 담보는 뺑소니 사고만 보상합니다."
    },
    {
        "query": "계약 취소하고 싶은데요",
        "positive": "보험 계약은 청약일로부터 15일 이내, 또는 증권을 받은 날로부터 15일 이내 취소 가능합니다.",
        "negative": "대물배상(임의)은 의무보험 한도를 초과한 손해를 보상합니다."
    },
    {
        "query": "주소 변경하려면 어떻게?",
        "positive": "주소 변경 등 중요 사실은 15일 이내에 회사에 알려야 하며, 미통지 시 불이익이 발생할 수 있습니다.",
        "negative": "자기신체사고는 피보험자 본인의 인적 손해만 보상합니다."
    },
]

MODEL_CANDIDATES = [
    "jhgan/ko-sroberta-multitask",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large"
]

PREFIX_CONFIG = {
    "e5": {"query": "query: ", "passage": "passage: "},
    "bge": {"query": "query: ", "passage": "passage: "}
}


def load_dataset() -> List[Dict[str, str]]:
    with open(BASE_DATASET_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)
    records: List[Dict[str, str]] = []
    for item in base:
        records.append({
            "query": item["query"],
            "positive": item["positive"],
            "negative": item.get("negative", "")
        })
    records.extend(ADDITIONAL_QUERIES)
    return records


def add_prefix(text: str, mode: str, model_name: str) -> str:
    lower = model_name.lower()
    for key, cfg in PREFIX_CONFIG.items():
        if key in lower:
            return cfg[mode] + text
    return text


def evaluate_model(model_name: str, dataset: List[Dict[str, str]]) -> Dict[str, float]:
    print(f"\n=== Evaluating {model_name} ===")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    pos_scores = []
    separations = []
    success_flags = []
    start_time = time.time()

    for entry in dataset:
        query = add_prefix(entry["query"], "query", model_name)
        positive = add_prefix(entry["positive"], "passage", model_name)
        negative = add_prefix(entry["negative"], "passage", model_name) if entry["negative"] else ""

        q_vec = embeddings.embed_query(query)
        p_vec = embeddings.embed_query(positive)

        pos_score = np.dot(q_vec, p_vec)
        pos_scores.append(pos_score)

        if negative:
            n_vec = embeddings.embed_query(negative)
            neg_score = np.dot(q_vec, n_vec)
            separation = pos_score - neg_score
            separations.append(separation)
            success_flags.append(1 if separation > 0 else 0)
        else:
            separations.append(np.nan)
            success_flags.append(np.nan)

    total_time = time.time() - start_time
    avg_time = total_time / len(dataset)

    result = {
        "model": model_name,
        "avg_pos": float(np.mean(pos_scores)),
        "avg_sep": float(np.nanmean(separations)),
        "recall": float(np.nanmean(success_flags)),
        "time_per_item": avg_time
    }
    return result


def main():
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} evaluation samples.")

    results = []
    for model in MODEL_CANDIDATES:
        try:
            res = evaluate_model(model, dataset)
            results.append(res)
        except Exception as exc:
            print(f"[ERROR] {model}: {exc}")

    print("\n=== Final Metrics ===")
    print(f"{'Model':<35} | {'Avg Pos':>8} | {'Avg Sep':>8} | {'Recall':>8} | {'Time(s)':>8}")
    print("-" * 80)
    for res in results:
        print(f"{res['model']:<35} | {res['avg_pos']:>8.4f} | {res['avg_sep']:>8.4f} | "
              f"{res['recall']:>8.4f} | {res['time_per_item']:>8.4f}")


if __name__ == "__main__":
    main()

