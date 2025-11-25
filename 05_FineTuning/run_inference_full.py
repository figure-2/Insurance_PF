import json
import torch
import os
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# 1. 설정 (Configuration)
BASE_MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
ADAPTER_MODEL_DIR = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora-v3"
TEST_DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/test_20.json"
OUTPUT_FILE = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/evaluation_result_full.json"

# 추론 설정
BATCH_SIZE = 8       # A100(40GB) 기준 8~16 정도면 안전하고 빠름
MAX_NEW_TOKENS = 512 # 생성할 답변의 최대 길이

def main():
    print(f"🚀 [Start] Full Inference on Test Set ({TEST_DATA_PATH})")

    # 2. 데이터 로드
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"📂 Loaded {len(test_data)} samples.")

    # 3. 모델 및 토크나이저 로드
    print("🤖 Loading Model...")
    
    # 4-bit 양자화 설정 (학습 때와 동일하게)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Base Model 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA Adapter 병합 (Inference Mode)
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_DIR)
    model.eval() # 평가 모드 전환

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # 생성 시에는 left padding이 국룰

    print("✅ Model Loaded Successfully!")

    # 4. 프롬프트 구성 및 배치 처리
    results = []
    
    # 배치 단위로 순회
    for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Processing Batches"):
        batch_items = test_data[i : i + BATCH_SIZE]
        
        # 프롬프트 구성 (Chat Template 적용 가능하지만 여기선 학습 포맷 유지)
        prompts = []
        for item in batch_items:
            # 학습 때와 동일한 포맷 사용 (답변 부분 제외)
            prompt = (
                f"### 지시\n{item['instruction']}\n\n"
                f"### 입력\n{item['input']}\n\n"
                f"### 출력\n" # 답변 생성을 유도하는 끝부분
            )
            prompts.append(prompt)

        # 토크나이징
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to("cuda")

        # 답변 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,    # 사실적인 답변을 위해 낮게 설정
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 디코딩 및 결과 저장
        # 입력 프롬프트 길이를 제외하고 생성된 텍스트만 추출
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for j, pred in enumerate(decoded_preds):
            original_item = batch_items[j]
            result_item = {
                "type": original_item.get("type", "unknown"),
                "instruction": original_item["instruction"],
                "ground_truth": original_item["output"], # 정답
                "prediction": pred.strip()               # 모델 예측
            }
            results.append(result_item)

    # 5. 결과 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Inference Completed! Results saved to {OUTPUT_FILE}")
    
    # 샘플 출력 (확인용)
    print("\n🔍 [Sample Result]")
    print(f"Q: {results[0]['instruction']}")
    print(f"A(Model): {results[0]['prediction']}")
    print(f"A(Truth): {results[0]['ground_truth']}")

if __name__ == "__main__":
    main()
