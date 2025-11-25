import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. 설정 (Configuration)
BASE_MODEL = "beomi/Llama-3-Open-Ko-8B"
# [v4] Stratified Split된 학습 데이터 사용
DATASET_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_v4.json"
# [v4] 출력 경로
OUTPUT_DIR = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora-v4"

# 하이퍼파라미터 (v4 데이터 규모 고려)
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

def main():
    print("🚀 [Start] QLoRA Fine-Tuning Phase 5 (v4 Dataset)")
    print(f"📌 Base Model: {BASE_MODEL}")
    print(f"📌 Dataset: {DATASET_PATH}")
    print(f"📌 Output Dir: {OUTPUT_DIR}")
    
    # 2. 데이터셋 로드 및 포맷팅
    print(f"📂 Loading Dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset file not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터셋 포맷팅 (Instruction Tuning Format)
    formatted_data = []
    for item in data:
        # v4 데이터의 필드 사용 (instruction, input, output)
        text = (
            f"### 지시\n{item['instruction']}\n\n"
            f"### 입력\n{item['input']}\n\n"
            f"### 출력\n{item['output']}<|end_of_text|>"
        )
        formatted_data.append({"text": text})

    # HuggingFace Dataset 변환
    dataset = Dataset.from_list(formatted_data)
    print(f"✅ Dataset Size: {len(dataset)} samples")

    # 포맷팅 함수
    def formatting_prompts_func(example):
        return example["text"]

    # 3. 모델 및 토크나이저 로드 (QLoRA 설정)
    print("🤖 Loading Model & Tokenizer...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA 설정
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. 학습 인자
    # 데이터가 늘어났으므로 save_steps를 조금 늘려줍니다 (200 -> 400)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=400,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )

    # 6. Trainer 설정
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
    )

    # 7. 학습 시작
    print("🔥 Training Start!")
    trainer.train()

    # 8. 모델 저장
    print(f"💾 Saving Model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Fine-Tuning Completed Successfully!")

if __name__ == "__main__":
    main()
