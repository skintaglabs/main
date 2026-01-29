# SkinTag: Multi-Dataset, Domain-Robust, Fairness-Aware Skin Lesion Triage

## Vision

An AI-powered screening tool for low-resource dermatological settings — helping people with limited access to dermatologists get early risk assessment from consumer phone photos. Not a diagnostic tool; a triage aid that tells users when to seek professional care urgently.

**Target venue**: NeurIPS (with explicit caveats about screening vs. diagnosis).

---

## Problem

1. **Domain bias**: Most dermatology AI is trained on dermoscopic images. Consumer phone photos look completely different. Models learn "dermoscope artifacts = pathological" instead of actual lesion features.
2. **Skin tone bias**: Training data is overwhelmingly Fitzpatrick skin types I-III (lighter skin). Models have lower sensitivity on types IV-VI (darker skin) — exactly the populations with worst access to dermatologists.
3. **Binary triage gap**: Users don't need a 114-class differential diagnosis. They need: "Is this urgent enough to see a doctor?"

## Approach

- **Multi-dataset training** across four complementary datasets spanning dermoscopic, clinical, and smartphone domains
- **Domain-bridging augmentations** that add/remove dermoscope artifacts so the model can't cheat
- **Fitzpatrick-balanced sampling** that explicitly upweights under-represented darker skin tones
- **Three modeling approaches** (naive baseline, classical ML, deep learning) for rigorous comparison
- **Triage output** with urgency tiers, not raw probabilities
- **Polished web app** for consumer use with prominent medical disclaimers

---

## Datasets

| Dataset | Images | Domain | Fitzpatrick | Labels | Source URL |
|---------|--------|--------|-------------|--------|------------|
| **HAM10000** | 10,015 | Dermoscopic | No | 7 classes → binary | [Kaggle](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset) |
| **DDI** (Stanford) | 656 | Clinical | Yes (grouped I-II, III-IV, V-VI) | Biopsy-proven benign/malignant | [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) |
| **Fitzpatrick17k** | ~16,577 | Clinical | Yes (1-6) | 114 conditions → three_partition_label (benign/malignant/non-neoplastic) | [GitHub](https://github.com/mattgroh/fitzpatrick17k) |
| **PAD-UFES-20** | 2,298 | Smartphone | Yes (via `fitspatrick` column) | 6 diagnostic categories → binary | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) |

### Why these four

- **HAM10000**: Large, well-labeled dermoscopic dataset. The baseline.
- **DDI**: Only dataset with balanced representation across all Fitzpatrick types. Gold standard for fairness evaluation. Biopsy-proven labels.
- **Fitzpatrick17k**: Adds volume of clinical images with per-image Fitzpatrick annotation. The `three_partition_label` column provides reliable binary mapping without fuzzy condition name matching.
- **PAD-UFES-20**: The critical smartphone-domain dataset. Breaks the dermoscope-pathology correlation. Also provides Fitzpatrick annotations, age, gender, and rich clinical metadata.

### Dataset CSV Column Reference

**HAM10000** (`HAM10000_metadata.csv`):
- `image_id`, `dx` (diagnosis), `dx_type`, `age`, `sex`, `localization`

**DDI** (`ddi_metadata.csv`):
- `DDI_file` (image filename), `skin_tone` (values: 12, 34, 56 = grouped FST), `malignancy(malig=1)` (0/1)

**Fitzpatrick17k** (`fitzpatrick17k.csv`):
- `md5hash` (image identifier/filename), `label` (114 conditions), `three_partition_label` (benign/malignant/non-neoplastic), `fitzpatrick` (1-6)

**PAD-UFES-20** (`metadata.csv`):
- `img_id`, `diagnostic` (ACK/BCC/MEL/NEV/SCC/SEK), `fitspatrick` (note: misspelled in original), `age`, `gender`, `region`

### Local Data Layout

```
data/
├── HAM10000_metadata.csv          # or data/Skin Cancer/...
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── ddi/
│   ├── ddi_metadata.csv
│   └── images/
├── fitzpatrick17k/
│   ├── fitzpatrick17k.csv
│   └── images/
└── pad_ufes/
    ├── metadata.csv
    └── images/
```

---

## Three Required Models

| Model | Type | Architecture | How it works |
|-------|------|-------------|-------------|
| **Naive baseline** | Majority class | Always predicts "benign" | No training. Sets floor for accuracy. |
| **Classical ML** | Logistic Regression | StandardScaler → LogisticRegression on 1152-d SigLIP embeddings | sklearn, sub-second training |
| **Deep learning** | Fine-tuned MLP head | 2-layer MLP (1152→256→2) with dropout+BatchNorm on SigLIP embeddings | PyTorch, early stopping, class-weighted loss |
| **End-to-end** (optional) | Fine-tuned SigLIP | Unfreezes last N transformer layers + classification head | GPU required, best accuracy for deployment |

The **deployed model** will be whichever performs best on the cross-domain evaluation, ranked by **F1 macro** (the primary metric for imbalanced dermatology data).

---

## Focused Experiment: Domain Shift & Fairness Sensitivity

**Question**: Does training with domain-bridging augmentations and multi-dataset balancing improve cross-domain generalization and fairness across skin tones?

**Protocol**:
1. Train all 3 models on HAM10000 only (baseline condition)
2. Train all 3 models on multi-dataset (no augmentation, no balancing)
3. Train all 3 models on multi-dataset with domain+Fitzpatrick balanced weights
4. For each: evaluate per-Fitzpatrick-type F1, per-domain F1, fairness gaps

**Primary metrics**: F1 macro, F1 (malignant), per-Fitzpatrick-type sensitivity, equalized odds gap, cross-domain F1 gap

---

## Fairness & Skin Tone Analysis

### The Problem

Dermatology datasets are overwhelmingly light-skinned:
- HAM10000: No Fitzpatrick annotations at all
- Fitzpatrick17k: Skewed toward types I-III
- Most clinical training data: 80%+ Fitzpatrick I-III

This means models trained naively will have significantly lower sensitivity (miss more cancers) on darker skin tones — precisely the populations with worst access to dermatologists.

### Our Approach

1. **Fitzpatrick-balanced sampling**: Upweights under-represented (Fitzpatrick type, label) pairs so each skin tone contributes equally to training loss
2. **DDI as fairness benchmark**: The only dataset with deliberate balanced representation across all skin tone groups
3. **Per-Fitzpatrick evaluation**: Report F1, sensitivity, specificity, and AUC broken down by Fitzpatrick type
4. **Equalized odds gap**: Measure the maximum difference in sensitivity and specificity across Fitzpatrick groups — the key fairness metric

---

## Architecture

```
data/                              # Local datasets (gitignored)
configs/config.yaml                # All configuration
src/
├── data/
│   ├── schema.py                  # SkinSample dataclass (unified schema)
│   ├── loader.py                  # load_multi_dataset() orchestrator
│   ├── datasets/                  # Per-dataset adapters (HAM10000, DDI, Fitz17k, PAD-UFES)
│   ├── sampler.py                 # Domain + Fitzpatrick balanced weights
│   ├── augmentations.py           # Training/eval transforms + domain bridging
│   └── dermoscope_aug.py          # Custom dermoscope artifact augmentations
├── model/
│   ├── embeddings.py              # SigLIP embedding extraction + caching
│   ├── classifier.py              # SklearnClassifier (logistic/MLP)
│   ├── baseline.py                # MajorityClass + RandomWeighted baselines
│   ├── deep_classifier.py         # DeepClassifier (head) + EndToEndClassifier
│   └── triage.py                  # TriageSystem with urgency tiers
├── evaluation/
│   └── metrics.py                 # F1, accuracy, AUC, per-group, equalized odds
scripts/
├── train.py                       # Main training (supports --multi-dataset --model all)
├── train_all_models.py            # Train + compare all 3 models
├── evaluate.py                    # Full fairness evaluation
└── evaluate_cross_domain.py       # Leave-one-domain-out experiment
app/
├── main.py                        # FastAPI backend
├── templates/index.html           # Polished frontend
└── static/                        # Static assets
Dockerfile                         # Containerized deployment
```

---

## Usage

### Setup
```bash
make install                       # Install dependencies
make data                          # Download HAM10000
# Manually download DDI, Fitzpatrick17k, PAD-UFES (see Makefile for URLs)
```

### Training
```bash
# Single dataset (HAM10000), logistic regression
python scripts/train.py

# Multi-dataset, all models, domain+Fitzpatrick balanced
python scripts/train.py --multi-dataset --domain-balance --model all

# Train + compare all 3 model types
python scripts/train_all_models.py

# Quick smoke test
python scripts/train.py --sample 500
```

### Evaluation
```bash
# Full fairness report
python scripts/evaluate.py --models logistic deep baseline

# Cross-domain experiment
python scripts/evaluate_cross_domain.py
```

### Web App
```bash
# Local development
make app                           # http://localhost:8000

# Docker
make app-docker
```

---

## Key Design Decisions

1. **F1 macro as primary metric** — accuracy is misleading on class-imbalanced data (70% benign → 70% accuracy by always predicting benign). F1 macro treats both classes equally.

2. **Fitzpatrick-balanced sampling, not oversampling** — oversampling duplicates minority images (overfitting risk). Balanced sample weights upweight minority samples in the loss function without duplication.

3. **Domain-bridging augmentations** — randomly add dermoscope artifacts to phone photos and remove them from dermoscopic images. The model sees all label-domain combinations, breaking spurious correlations.

4. **three_partition_label for Fitzpatrick17k** — the dataset provides a reliable benign/malignant/non-neoplastic partition. Using this instead of fuzzy string matching against 114 condition names eliminates misclassification risk.

5. **DDI skin_tone as grouped Fitzpatrick** — DDI uses groups (I-II, III-IV, V-VI) not individual types. We map to midpoints (1, 3, 5) for consistency but note the grouping in analysis.

6. **Triage tiers, not probabilities** — users don't understand "0.47 probability of malignancy." They understand "moderate concern — schedule a dermatology appointment within 2-4 weeks."

7. **Medical disclaimer everywhere** — this is a screening aid, not a diagnosis. Every output includes a prominent disclaimer. The app makes this impossible to miss.
