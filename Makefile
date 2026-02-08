.PHONY: help venv install-gpu data data-ddi data-pad-ufes pipeline pipeline-quick train train-all train-multi evaluate evaluate-cross-domain app preview stop clean

# Python interpreter (prefers venv if available)
PYTHON := $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)
PYTHON_ENV := OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
PORT := 8000

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "  LOCAL - CPU (inference only)"
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "Setup:"
	@echo "  venv               Create venv + install dependencies"
	@echo ""
	@echo "Application:"
	@echo "  app                Start inference API server (port $(PORT))"
	@echo "  preview            Start React dev server for webapp preview"
	@echo "  stop               Stop running servers"
	@echo "  clean              Remove cached embeddings and models"
	@echo ""
	@echo "Evaluation (pre-trained models):"
	@echo "  evaluate           Fairness evaluation on test set"
	@echo "  evaluate-cross-domain  Cross-domain generalization"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "  LOCAL - NVIDIA GPU (training)"
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "Setup:"
	@echo "  install-gpu        Install dependencies with CUDA support"
	@echo ""
	@echo "Data:"
	@echo "  data               Download HAM10000 dataset from Kaggle"
	@echo "  data-ddi           Download DDI dataset (Stanford AIMI access)"
	@echo "  data-pad-ufes      Download PAD-UFES-20 dataset"
	@echo ""
	@echo "Training:"
	@echo "  pipeline           Full pipeline: data → embed → train → eval"
	@echo "  train              Train logistic regression (HAM10000)"
	@echo "  train-all          Train all 3 models (baseline, logistic, deep)"
	@echo "  train-multi        Train with multi-dataset + domain balancing"
	@echo ""

# ════════════════════════════════════════════════════════════════════
# LOCAL - CPU (inference only)
# ════════════════════════════════════════════════════════════════════

venv:
	@if [ -d venv ]; then \
		echo "Virtual environment already exists at ./venv"; \
	else \
		echo "Creating virtual environment..."; \
		if command -v python3.11 >/dev/null 2>&1; then \
			python3.11 -m venv venv; \
		else \
			python3 -m venv venv; \
		fi; \
		echo "Installing dependencies..."; \
		venv/bin/pip install --upgrade pip; \
		venv/bin/pip install -r requirements.txt; \
		echo "Virtual environment created and dependencies installed"; \
		echo "Run 'make app' or other commands - they will automatically use the venv"; \
	fi

app:
	$(PYTHON_ENV) $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

preview:
	@API_URL=$$(./scripts/get_api_url.sh); \
	echo "Using API URL: $$API_URL"; \
	cd webapp-react && VITE_API_URL=$$API_URL npm install && VITE_API_URL=$$API_URL npm run dev

stop:
	@pkill -f "uvicorn app.main:app" 2>/dev/null || true
	@lsof -ti:$(PORT) | xargs kill -9 2>/dev/null || true
	@echo "Stopped"

clean:
	rm -rf results/cache/*

evaluate:
	$(PYTHON_ENV) $(PYTHON) scripts/evaluate.py --models logistic deep baseline

evaluate-cross-domain:
	$(PYTHON_ENV) $(PYTHON) scripts/evaluate_cross_domain.py

# ════════════════════════════════════════════════════════════════════
# LOCAL - NVIDIA GPU (training)
# ════════════════════════════════════════════════════════════════════

install-gpu:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

data:
	pip install -q kaggle
	mkdir -p data
	kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset -p data/ --unzip

data-ddi:
	@echo "DDI dataset requires Stanford AIMI access."
	@echo "1. Visit https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965"
	@echo "2. Download and extract to data/ddi/"
	@echo "   Expected: data/ddi/ddi_metadata.csv and data/ddi/images/"
	mkdir -p data/ddi/images

data-pad-ufes:
	@echo "PAD-UFES-20 dataset download:"
	@echo "1. Visit https://data.mendeley.com/datasets/zr7vgbcyr2/1"
	@echo "2. Download and extract to data/pad_ufes/"
	@echo "   Expected: data/pad_ufes/metadata.csv and data/pad_ufes/images/"
	mkdir -p data/pad_ufes/images

pipeline:
	$(PYTHON_ENV) $(PYTHON) run_pipeline.py

pipeline-quick:
	$(PYTHON_ENV) $(PYTHON) run_pipeline.py --quick --no-app

train:
	$(PYTHON_ENV) $(PYTHON) scripts/train.py

train-all:
	$(PYTHON_ENV) $(PYTHON) scripts/train_all_models.py

train-multi:
	$(PYTHON_ENV) $(PYTHON) scripts/train.py --multi-dataset --domain-balance --model all
