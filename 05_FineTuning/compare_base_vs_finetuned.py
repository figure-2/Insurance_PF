import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm

# 설정
BASE_MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
ADAPTER_MODEL_DIR = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora-v3"
TEST_DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/test_20.json"
OUTPUT_FILE = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/comparison_base_vs_finetuned.json"

# 샘플링: 전체 1,088개 중 50개만 비교 (빠른 확인용)
SAMPLE_SIZE = 50
MAX_NEW_TOKENS = 512

def format_prompt(instruction, input_text=""):
    """학습 포맷과 동일하게 프롬프트 구성"""
    if input_text:
        return f"### 지시\n{instruction}\n\n### 입력\n{input_text}\n\n### 출력\n"
    else:
        return f"### 지시\n{instruction}\n\n### 입력\n{input_text}\n\n### 출력\n"

def generate_answer(model, tokenizer, prompt):
    """모델 답변 생성"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_len:]
    answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    return answer

def main():
    print("🚀 [Start] Base Model vs Fine-Tuned Model Comparison")
    
    # 데이터 로드 및 샘플링
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 랜덤 샘플링 (재현성을 위해 시드 고정)
    import random
    random.seed(42)
    sampled_data = random.sample(test_data, min(SAMPLE_SIZE, len(test_data)))
    
    print(f"📂 Loaded {len(test_data)} samples, sampling {len(sampled_data)} for comparison.")
    
    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 1. Base Model 로드
    print("\n🤖 [1/2] Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = "left"
    print("✅ Base Model Loaded!")
    
    # 2. Fine-Tuned Model 로드
    print("\n🤖 [2/2] Loading Fine-Tuned Model...")
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_DIR)
    finetuned_model.eval()
    finetuned_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
    finetuned_tokenizer.padding_side = "left"
    print("✅ Fine-Tuned Model Loaded!")
    
    # 3. 비교 추론
    print(f"\n🔄 Generating answers for {len(sampled_data)} samples...")
    results = []
    
    for item in tqdm(sampled_data, desc="Comparing"):
        prompt = format_prompt(item['instruction'], item['input'])
        
        # Base Model 답변
        base_answer = generate_answer(base_model, base_tokenizer, prompt)
        
        # Fine-Tuned Model 답변
        finetuned_answer = generate_answer(finetuned_model, finetuned_tokenizer, prompt)
        
        result = {
            "type": item.get("type", "unknown"),
            "instruction": item["instruction"],
            "input": item["input"],
            "ground_truth": item["output"],
            "base_model_answer": base_answer,
            "finetuned_model_answer": finetuned_answer,
        }
        results.append(result)
    
    # 4. 결과 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Comparison results saved to {OUTPUT_FILE}")
    
    # 5. 샘플 출력
    print("\n" + "=" * 80)
    print("🔍 [Sample Comparison]")
    print("=" * 80)
    sample = results[0]
    print(f"\n📋 Q: {sample['instruction'][:100]}...")
    print(f"\n✅ 정답: {sample['ground_truth'][:150]}...")
    print(f"\n🔵 Base Model: {sample['base_model_answer'][:150]}...")
    print(f"\n🟢 Fine-Tuned: {sample['finetuned_model_answer'][:150]}...")
    
    print("\n✅ Comparison Completed!")

if __name__ == "__main__":
    main()

