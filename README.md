# SkinTag

Robust skin lesion classification using MedSigLIP embeddings with augmentations targeting demographic and environmental variations.

## Approach

1. **Pre-trained Model**: MedSigLIP (400M vision encoder trained on dermatology images)
2. **Transfer Learning**: Extract embeddings, train lightweight classifier
3. **Augmentations**: Skin tone simulation, lighting variation, image noise

## Setup

```bash
pip install -r requirements.txt

# Download HAM10000 dataset from Kaggle
pip install kaggle
kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset -p data/ --unzip
```

## Usage

### Local

```bash
make install   # Install dependencies
make data      # Download HAM10000 dataset
make train     # Run training
make evaluate  # Run evaluation
```

### GitHub Actions

1. Go to **Actions → Train Model → Run workflow**
2. Enter `kaggle://farjanakabirsamanta/skin-cancer-dataset` as dataset URL
3. Results appear in the workflow summary

## Structure

```
├── notebooks/demo.ipynb    # Presentation notebook
├── src/
│   ├── data/               # Data loading and augmentations
│   ├── model/              # MedSigLIP embeddings + classifier
│   └── evaluation/         # Robustness metrics
├── configs/config.yaml     # Hyperparameters
└── results/                # Figures and metrics
```
