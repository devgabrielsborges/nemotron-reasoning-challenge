import random
import os

import kagglehub
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from nemotron.config import NemotronConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_path(cfg: NemotronConfig) -> str:
    if cfg.model_path:
        return cfg.model_path
    return kagglehub.model_download(cfg.kaggle_model_handle)


def build_training_text(tokenizer, prompt: str, answer: str) -> str:
    user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."
    assistant_msg = f"\\boxed{{{answer}}}"

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )


class SFTDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.examples = []
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main() -> None:
    load_dotenv(override=True)
    cfg = NemotronConfig.from_env()

    if cfg.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

    set_seed(cfg.random_seed)

    if not cfg.train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {cfg.train_csv}")

    train_df = pd.read_csv(cfg.train_csv)
    required_cols = {cfg.prompt_column, cfg.target_column}
    missing = required_cols - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in train data: {sorted(missing)}")

    if cfg.subsample_size and cfg.subsample_size < len(train_df):
        train_df = train_df.sample(n=cfg.subsample_size, random_state=cfg.random_seed)

    model_path = resolve_model_path(cfg)
    print(f"Using model path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        if cfg.use_cpu_offload:
            cfg.offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(cfg.offload_dir)
            model_kwargs["max_memory"] = {
                0: f"{cfg.gpu_max_memory_gib}GiB",
                "cpu": f"{cfg.cpu_max_memory_gib}GiB",
            }

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_targets = [
        item.strip() for item in cfg.lora_target_modules.split(",") if item.strip()
    ]
    if not lora_targets:
        raise ValueError("NEMOTRON_LORA_TARGET_MODULES cannot be empty.")

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    texts = [
        build_training_text(
            tokenizer, str(row[cfg.prompt_column]), str(row[cfg.target_column])
        )
        for _, row in train_df.iterrows()
    ]
    dataset = SFTDataset(texts, tokenizer, cfg.max_seq_len)
    print(f"Prepared {len(dataset)} training examples")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir / "trainer_output"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        report_to=[],
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_pin_memory=False,
        optim=cfg.optimizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved adapter to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
