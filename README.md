# NVIDIA Nemotron Model Reasoning Challenge Starter

This repository is adapted for the Kaggle **NVIDIA Nemotron Model Reasoning Challenge**.

The competition expects a `submission.zip` containing a compatible LoRA adapter for **Nemotron-3-Nano-30B**, including `adapter_config.json`, with **LoRA rank <= 32**.

## What is implemented

- `src/nemotron/train_lora.py`: supervised LoRA fine-tuning pipeline based on `prompt` + `answer` data.
- `src/nemotron/package_submission.py`: packages adapter files into competition-ready `submission.zip` and validates `adapter_config.json`.
- `src/nemotron/config.py`: centralized environment-driven configuration with competition guardrails.

The previous scikit-learn/MLflow template remains available under `src/models` for experimentation, but it is **not** the primary submission path for this challenge.

## Quick Start

### 1) Configure env

```bash
cp .env.example .env
```

Set or confirm these key values in `.env`:

- `KAGGLE_COMPETITION_NAME=nvidia-nemotron-model-reasoning-challenge`
- `TARGET_COLUMN=answer`
- `PROMPT_COLUMN=prompt`
- `NEMOTRON_LORA_RANK=32` (must be <= 32)
- Optional `NEMOTRON_MODEL_PATH=/path/to/local/model` (otherwise `kagglehub` downloads)

### 2) Install dependencies

```bash
uv sync
```

### 3) Download data

```bash
make init
```

### 4) Train LoRA adapter

```bash
make nemotron-train
```

Artifacts are written to `NEMOTRON_OUTPUT_DIR` (default: `artifacts/adapter`).

### 5) Create submission zip

```bash
make nemotron-package
```

This creates `NEMOTRON_SUBMISSION_PATH` (default: `artifacts/submission.zip`) and verifies required adapter metadata.

### 6) One-command run

```bash
make nemotron-all
```

## Nemotron environment variables

Defined in `.env.example`:

- Data/model paths: `NEMOTRON_TRAIN_CSV`, `NEMOTRON_MODEL_PATH`, `NEMOTRON_KAGGLE_MODEL_HANDLE`
- Output paths: `NEMOTRON_OUTPUT_DIR`, `NEMOTRON_SUBMISSION_PATH`
- Training: `NEMOTRON_MAX_SEQ_LEN`, `NEMOTRON_NUM_EPOCHS`, `NEMOTRON_GRAD_ACCUM`, `NEMOTRON_LR`, `NEMOTRON_BATCH_SIZE`
- LoRA: `NEMOTRON_LORA_RANK`, `NEMOTRON_LORA_ALPHA`, `NEMOTRON_LORA_DROPOUT`
- Optional dev speed: `NEMOTRON_SUBSAMPLE_SIZE`

## Make targets

```bash
make help
make init
make nemotron-train
make nemotron-package
make nemotron-all
```
