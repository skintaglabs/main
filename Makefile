.PHONY: help install data train evaluate clean

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install dependencies"
	@echo "  data      Download HAM10000 dataset from Kaggle"
	@echo "  train     Run training"
	@echo "  evaluate  Run evaluation"
	@echo "  clean     Remove cached embeddings and models"

install:
	pip install -r requirements.txt

data:
	pip install -q kaggle
	mkdir -p data
	kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset -p data/ --unzip

train:
	PYTHONPATH=. python scripts/train.py

evaluate:
	PYTHONPATH=. python scripts/evaluate.py

clean:
	rm -rf results/cache/*
