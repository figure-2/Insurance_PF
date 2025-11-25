"""
Compare 3 Models:
1. Base Model (No Context)
2. Fine-Tuned Model v3 (No Context)
3. RAG + Fine-Tuned Model v3 (With Context)
"""

import json
import torch
import random
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 프로젝트 루트 경로 설정
sys.path.append("/home/pencilfoxs/0_Insurance_PF/03_Retrieval")
from bm25_retriever import MecabBM25Retriever

# 설정
BASE_MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
ADAPTER_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora-v3"
TEST_DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/test_20.json"
OUTPUT_FILE = "/home/pencilfoxs/0_Insurance_PF/06_Evaluation/comparison_3_models.json"
SAMPLE_SIZE = 30

def load_resources():
    """모든 리소스 로드"""
    print("Loading resources...")
    
    # 1. Retriever
    retriever = MecabBM25Retriever.load_index(
        "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    )
    retriever.k = 3
    
    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model (Base + Adapter)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Base Model 로드
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # LoRA 어댑터 결합
    print(f"Loading LoRA Adapter: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    return model, tokenizer, retriever

def generate(model, tokenizer, prompt):
    """답변 생성"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 부분 제거
    if "### 출력" in text:
        return text.split("### 출력")[1].strip()
    elif "### 답변" in text:
        return text.split("### 답변")[1].strip()
    
    # 프롬프트 길이만큼 잘라내기 (fallback)
    prompt_len = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    return text[prompt_len:].strip()

def build_prompt(instruction, input_text=""):
    """학습 데이터 포맷과 일치하는 프롬프트 생성"""
    if input_text:
        return f"### 지시\n{instruction}\n\n### 입력\n{input_text}\n\n### 출력\n"
    else:
        return f"### 지시\n{instruction}\n\n### 입력\n\n### 출력\n"

def main():
    model, tokenizer, retriever = load_resources()
    
    # 테스트 데이터 로드
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 샘플링
    random.seed(42)
    samples = random.sample(data, min(SAMPLE_SIZE, len(data)))
    
    results = []
    
    print(f"Starting comparison on {len(samples)} samples...")
    
    for item in tqdm(samples):
        question = item['instruction']
        ground_truth = item['output']
        
        # 1. Base Model (Adapter Disabled)
        with model.disable_adapter():
            base_prompt = build_prompt(question, input_text="")
            base_answer = generate(model, tokenizer, base_prompt)
            
        # 2. Fine-Tuned Model (Adapter Enabled, No Context)
        ft_prompt = build_prompt(question, input_text="")
        ft_answer = generate(model, tokenizer, ft_prompt)
        
        # 3. RAG + Fine-Tuned (Adapter Enabled, With Context)
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        rag_prompt = build_prompt(question, input_text=context)
        rag_answer = generate(model, tokenizer, rag_prompt)
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "base_model_answer": base_answer,
            "finetuned_model_answer": ft_answer,
            "rag_finetuned_answer": rag_answer,
            "retrieved_context": context[:500] + "..." if len(context) > 500 else context
        })
        
    # 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\nComparison completed! Results saved to {OUTPUT_FILE}")
    print(f"Total samples: {len(results)}")

if __name__ == "__main__":
    main()

