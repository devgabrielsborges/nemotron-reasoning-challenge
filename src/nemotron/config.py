import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NemotronConfig:
    train_csv: Path
    prompt_column: str
    target_column: str
    model_path: str | None
    kaggle_model_handle: str
    output_dir: Path
    submission_zip_path: Path
    subsample_size: int | None
    random_seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    max_seq_len: int
    num_epochs: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    optimizer: str
    use_4bit: bool
    bnb_compute_dtype: str
    bnb_quant_type: str
    bnb_use_double_quant: bool
    cpu_only: bool
    cpu_dtype: str
    cuda_visible_devices: str | None
    gpu_max_memory_gib: int
    cpu_max_memory_gib: int
    use_cpu_offload: bool
    offload_dir: Path

    @classmethod
    def from_env(cls) -> "NemotronConfig":
        data_raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
        output_dir = Path(os.getenv("NEMOTRON_OUTPUT_DIR", "artifacts/adapter"))
        submission_zip_path = Path(
            os.getenv("NEMOTRON_SUBMISSION_PATH", "artifacts/submission.zip")
        )

        subsample_raw = os.getenv("NEMOTRON_SUBSAMPLE_SIZE", "")
        subsample_size = int(subsample_raw) if subsample_raw else None

        lora_rank = int(os.getenv("NEMOTRON_LORA_RANK", "32"))
        if lora_rank > 32:
            raise ValueError("NEMOTRON_LORA_RANK must be <= 32 for competition rules.")

        return cls(
            train_csv=Path(
                os.getenv("NEMOTRON_TRAIN_CSV", str(data_raw_dir / "train.csv"))
            ),
            prompt_column=os.getenv("PROMPT_COLUMN", "prompt"),
            target_column=os.getenv("TARGET_COLUMN", "answer"),
            model_path=os.getenv("NEMOTRON_MODEL_PATH"),
            kaggle_model_handle=os.getenv(
                "NEMOTRON_KAGGLE_MODEL_HANDLE",
                "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            ),
            output_dir=output_dir,
            submission_zip_path=submission_zip_path,
            subsample_size=subsample_size,
            random_seed=int(os.getenv("NEMOTRON_RANDOM_SEED", "42")),
            lora_rank=lora_rank,
            lora_alpha=int(os.getenv("NEMOTRON_LORA_ALPHA", "16")),
            lora_dropout=float(os.getenv("NEMOTRON_LORA_DROPOUT", "0.05")),
            lora_target_modules=os.getenv("NEMOTRON_LORA_TARGET_MODULES", "out_proj"),
            max_seq_len=int(os.getenv("NEMOTRON_MAX_SEQ_LEN", "1024")),
            num_epochs=int(os.getenv("NEMOTRON_NUM_EPOCHS", "1")),
            grad_accum_steps=int(os.getenv("NEMOTRON_GRAD_ACCUM", "4")),
            learning_rate=float(os.getenv("NEMOTRON_LR", "2e-4")),
            weight_decay=float(os.getenv("NEMOTRON_WEIGHT_DECAY", "0.01")),
            batch_size=int(os.getenv("NEMOTRON_BATCH_SIZE", "1")),
            warmup_ratio=float(os.getenv("NEMOTRON_WARMUP_RATIO", "0.03")),
            logging_steps=int(os.getenv("NEMOTRON_LOGGING_STEPS", "10")),
            save_steps=int(os.getenv("NEMOTRON_SAVE_STEPS", "200")),
            optimizer=os.getenv("NEMOTRON_OPTIMIZER", "adafactor"),
            use_4bit=os.getenv("NEMOTRON_USE_4BIT", "true").lower()
            in ("true", "1", "yes"),
            bnb_compute_dtype=os.getenv("NEMOTRON_BNB_COMPUTE_DTYPE", "bfloat16"),
            bnb_quant_type=os.getenv("NEMOTRON_BNB_QUANT_TYPE", "nf4"),
            bnb_use_double_quant=os.getenv(
                "NEMOTRON_BNB_USE_DOUBLE_QUANT", "true"
            ).lower()
            in ("true", "1", "yes"),
            cpu_only=os.getenv("NEMOTRON_CPU_ONLY", "false").lower()
            in ("true", "1", "yes"),
            cpu_dtype=os.getenv("NEMOTRON_CPU_DTYPE", "bfloat16"),
            cuda_visible_devices=os.getenv("NEMOTRON_CUDA_VISIBLE_DEVICES", "0"),
            gpu_max_memory_gib=int(os.getenv("NEMOTRON_GPU_MAX_MEMORY_GIB", "20")),
            cpu_max_memory_gib=int(os.getenv("NEMOTRON_CPU_MAX_MEMORY_GIB", "120")),
            use_cpu_offload=os.getenv("NEMOTRON_USE_CPU_OFFLOAD", "true").lower()
            in ("true", "1", "yes"),
            offload_dir=Path(os.getenv("NEMOTRON_OFFLOAD_DIR", "artifacts/offload")),
        )
