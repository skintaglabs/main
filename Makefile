.PHONY: help install data data-ddi data-pad-ufes train train-all train-multi evaluate evaluate-cross-domain app app-docker clean

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install            Install dependencies"
	@echo "  data               Download HAM10000 dataset from Kaggle"
	@echo "  data-ddi           Download DDI dataset (requires Stanford AIMI access)"
	@echo "  data-pad-ufes      Download PAD-UFES-20 dataset"
	@echo ""
	@echo "Training:"
	@echo "  train              Train logistic regression classifier (HAM10000 only)"
	@echo "  train-all          Train all 3 model types (baseline, logistic, deep)"
	@echo "  train-multi        Train with multi-dataset + domain balancing"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluate           Run fairness evaluation on test set"
	@echo "  evaluate-cross-domain  Run cross-domain generalization experiment"
	@echo ""
	@echo "Application:"
	@echo "  app                Run web app locally"
	@echo "  app-docker         Build and run web app in Docker"
	@echo ""
	@echo "  clean              Remove cached embeddings and models"

install:
	pip install -r requirements.txt

# Dataset downloads
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

# Training
train:
	PYTHONPATH=. python scripts/train.py

train-all:
	PYTHONPATH=. python scripts/train_all_models.py

train-multi:
	PYTHONPATH=. python scripts/train.py --multi-dataset --domain-balance --model all

# Evaluation
evaluate:
	PYTHONPATH=. python scripts/evaluate.py --models logistic deep baseline

evaluate-cross-domain:
	PYTHONPATH=. python scripts/evaluate_cross_domain.py

# Application
app:
	PYTHONPATH=. python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

app-docker:
	docker build -t skintag .
	docker run -p 8000:8000 -v $(PWD)/results:/app/results -v $(PWD)/data:/app/data skintag

clean:
	rm -rf results/cache/*
