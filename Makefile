MODELS := logistic_regression random_forest svm xgboost_ gradient_boosting knn

.PHONY: help up down restart logs preprocess download train-% train-all clean

help:
	@echo "Infrastructure"
	@echo "  make up             Start Postgres + MinIO + MLflow"
	@echo "  make down           Stop all services"
	@echo "  make restart        Restart all services"
	@echo "  make logs           Tail service logs"
	@echo "  make clean          Stop services and delete volumes"
	@echo ""
	@echo "Data"
	@echo "  make download       Download dataset from Kaggle"
	@echo "  make preprocess     Run preprocessing pipeline"
	@echo ""
	@echo "Training"
	@echo "  make train-<model>  Train a single model (e.g. make train-random_forest)"
	@echo "  make train-all      Train all models sequentially"
	@echo ""
	@echo "Nemotron Competition"
	@echo "  make nemotron-train    Train LoRA adapter for Nemotron"
	@echo "  make nemotron-train-lowmem  Low-memory training profile"
	@echo "  make nemotron-package  Build submission.zip from adapter"
	@echo "  make nemotron-all      Train + package submission"
	@echo ""
	@echo "Available models: $(MODELS)"

up:
	docker compose up -d --build
	@echo ""
	@echo "MLflow UI:      http://localhost:5000"
	@echo "MinIO Console:  http://localhost:9001  (minioadmin / minioadmin)"

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

init:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/utils/download_dataset.py
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/preprocessing/preprocess.py

train-%:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/models/$*.py

train-all:
	@for model in $(MODELS); do \
		echo "\n========== Training $$model =========="; \
		set -a && [ -f .env ] && . ./.env && set +a; \
		uv run --python 3.11 src/models/$$model.py; \
	done

nemotron-train:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	CUDA_VISIBLE_DEVICES=$${NEMOTRON_CUDA_VISIBLE_DEVICES:-$${CUDA_VISIBLE_DEVICES:-0,1}} \
	PYTORCH_CUDA_ALLOC_CONF=$${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64} \
	PYTHONPATH=src uv run --python 3.11 -m nemotron.train_lora

nemotron-train-lowmem:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	NEMOTRON_LORA_RANK=$${NEMOTRON_LORA_RANK:-4} \
	NEMOTRON_LORA_TARGET_MODULES=$${NEMOTRON_LORA_TARGET_MODULES:-out_proj} \
	NEMOTRON_MAX_SEQ_LEN=$${NEMOTRON_MAX_SEQ_LEN:-256} \
	NEMOTRON_GRAD_ACCUM=$${NEMOTRON_GRAD_ACCUM:-8} \
	NEMOTRON_SUBSAMPLE_SIZE=$${NEMOTRON_SUBSAMPLE_SIZE:-200} \
	NEMOTRON_NUM_EPOCHS=$${NEMOTRON_NUM_EPOCHS:-1} \
	NEMOTRON_OPTIMIZER=$${NEMOTRON_OPTIMIZER:-adafactor} \
	CUDA_VISIBLE_DEVICES=$${NEMOTRON_CUDA_VISIBLE_DEVICES:-$${CUDA_VISIBLE_DEVICES:-0,1}} \
	PYTORCH_CUDA_ALLOC_CONF=$${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64} \
	PYTHONPATH=src uv run --python 3.11 -m nemotron.train_lora

nemotron-package:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	PYTHONPATH=src uv run --python 3.11 -m nemotron.package_submission

nemotron-all: nemotron-train-lowmem nemotron-package

clean:
	docker compose down -v
