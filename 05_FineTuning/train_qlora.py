"""
QLoRA 파인튜닝 스크립트
4-bit 양자화된 모델에 LoRA 어댑터를 학습시켜 메모리 효율적으로 파인튜닝
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer

# 설정
MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
DATASET_PATH = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/train_dataset.json"
OUTPUT_DIR = "/home/pencilfoxs/0_Insurance_PF/05_FineTuning/llama-3-ko-insurance-lora"


def load_model_and_tokenizer():
    """4-bit 양자화 모델과 토크나이저 로드"""
    print(f"Loading base model: {MODEL_ID}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # k-bit 학습을 위한 준비
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def setup_lora(model):
    """LoRA 설정 및 적용"""
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha (scaling factor)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama 모델의 attention 레이어
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def load_training_dataset():
    """학습 데이터셋 로드"""
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Loaded {len(dataset)} samples")
    return dataset


def train():
    """QLoRA 파인튜닝 실행"""
    # 1. 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. LoRA 설정
    model = setup_lora(model)
    
    # 3. 데이터셋 로드
    dataset = load_training_dataset()
    
    # 4. 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=200,  # 에폭 대신 스텝 수로 제어
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        optim="paged_adamw_32bit",  # 메모리 효율적인 옵티마이저
        warmup_steps=10,
        report_to="none"  # wandb 등 로깅 비활성화
    )
    
    # 5. Trainer 생성 및 학습
    # TRL 0.25.1 버전에 맞게 수정
    def formatting_func(example):
        return example["text"]
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func  # 데이터 포맷팅 함수 사용
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # 6. 모델 저장
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nTraining completed! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()

