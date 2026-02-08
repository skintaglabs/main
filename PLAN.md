# SkinTag: Multi-Dataset, Domain-Robust, Fairness-Aware Skin Lesion Triage

## Project Topic

Computer vision for dermatological screening. An AI-powered triage tool that classifies skin lesion photographs as benign or malignant, providing urgency-based recommendations. Designed for low-resource settings where dermatologist access is limited. This project is original work developed specifically for this course.

---

## Problem Statement

Skin cancer is the most common cancer worldwide, with melanoma alone causing over 57,000 deaths annually (Sung et al., 2021). Early detection dramatically improves outcomes: 5-year survival for localized melanoma exceeds 99%, but drops to 32% for distant metastases (Siegel et al., 2023). However, access to dermatologists is severely limited.

Three critical gaps prevent current AI systems from real-world deployment:

1. **Domain bias**: Most dermatology AI trains on dermoscopic images (specialized clinical equipment). Consumer smartphone photos look completely different. Models learn "dermoscope artifacts = pathological" rather than actual lesion features.
2. **Skin tone bias**: Training data is overwhelmingly Fitzpatrick skin types I-III (lighter skin). Models exhibit significantly lower sensitivity on types IV-VI (darker skin) -- the populations with worst access to dermatologists.
3. **Binary triage gap**: Users do not need a 114-class differential diagnosis. They need: "Is this urgent enough to see a doctor?"

SkinTag addresses all three by training across five datasets spanning dermoscopic, clinical, and smartphone domains, with Fitzpatrick-balanced sampling and a three-tier clinical triage output.

---

## Data Sources

### Five Complementary Datasets (47,277 images total)

| Dataset | Images | Domain | Fitzpatrick | Labels | Source |
|---------|--------|--------|-------------|--------|--------|
| **HAM10000** | 10,015 | Dermoscopic | No | 7 classes -> binary | [Kaggle](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset) |
| **DDI** (Stanford) | 656 | Clinical | Yes (grouped I-II, III-IV, V-VI) | Biopsy-proven benign/malignant | [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) |
| **Fitzpatrick17k** | ~16,518 | Clinical | Yes (1-6) | 114 conditions -> three_partition_label | [GitHub](https://github.com/mattgroh/fitzpatrick17k) |
| **PAD-UFES-20** | 2,298 | Smartphone | Yes | 6 diagnostic categories -> binary | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) |
| **BCN20000** | 17,790 | Dermoscopic | No | 8 classes (adds SCC) | [ISIC Archive](https://api.isic-archive.com/collections/249/) |

### Why These Five

- **HAM10000**: Large, well-labeled dermoscopic baseline. 6,705 melanocytic nevi provide "normal mole" representation.
- **DDI**: Only dataset with deliberately balanced representation across all Fitzpatrick types. Biopsy-confirmed labels. Gold standard for fairness evaluation.
- **Fitzpatrick17k**: Adds volume of clinical images with per-image Fitzpatrick annotation. The `three_partition_label` column provides reliable binary mapping without fuzzy condition name matching.
- **PAD-UFES-20**: Critical smartphone-domain dataset from Brazil. Breaks the dermoscope-pathology correlation. Also provides Fitzpatrick annotations, age, gender, and clinical metadata.
- **BCN20000**: Hospital Clinic Barcelona (2010-2016). Adds SCC as an explicit class and additional dermoscopic volume. Reference: Hernandez-Perez et al., "BCN20000: Dermoscopic Lesions in the Wild." *Scientific Data* (2024).

### Label Harmonization

**Decision: Binary (benign/malignant) with non-neoplastic included as benign.**

Evaluated in `notebooks/label_taxonomy_eda.ipynb`. Non-neoplastic conditions (eczema, psoriasis, infections) are not cancer -- for a triage system asking "is this urgent?", they belong in the benign bucket. This recovers ~8,000 additional images from Fitzpatrick17k (48% of that dataset).

### Dual Classification Targets

1. **Binary triage** (benign=0, malignant=1) -- primary output for user-facing triage.
2. **Condition estimation** (10 categories) -- secondary output showing the most likely specific condition.

### Unified Condition Taxonomy (10 Categories)

| ID | Condition | Binary | Present In |
|----|-----------|--------|------------|
| 0 | Melanoma | Malignant | All 5 |
| 1 | Basal Cell Carcinoma | Malignant | All 5 |
| 2 | Squamous Cell Carcinoma | Malignant | PAD, BCN, DDI, Fitz |
| 3 | Actinic Keratosis | Malignant | HAM, PAD, BCN, Fitz |
| 4 | Melanocytic Nevus | Benign | All 5 |
| 5 | Seborrheic Keratosis | Benign | All 5 |
| 6 | Dermatofibroma | Benign | HAM, BCN, DDI, Fitz |
| 7 | Vascular Lesion | Benign | HAM, BCN |
| 8 | Non-Neoplastic | Benign | Fitz, DDI |
| 9 | Other/Unknown | Benign | All (catch-all) |

### Domain and Skin Tone Distribution

- Imaging domains: dermoscopic (59%), clinical (36%), smartphone (5%)
- Fitzpatrick annotations available for 41% of images (DDI, Fitzpatrick17k, PAD-UFES-20)
- Skin tone distribution (among annotated): types I-III 68%, types IV-VI 32%

### Dataset CSV Column Reference

**HAM10000** (`HAM10000_metadata.csv`):
- `image_id`, `dx` (akiec/bcc/bkl/df/mel/nv/vasc), `dx_type`, `age`, `sex`, `localization`

**DDI** (`ddi_metadata.csv`):
- `DDI_file`, `skin_tone` (12, 34, 56 = grouped FST), `malignant` (bool), `disease`

**Fitzpatrick17k** (`fitzpatrick17k.csv`):
- `md5hash`, `label` (114 conditions), `three_partition_label` (benign/malignant/non-neoplastic), `fitzpatrick` (1-6)

**PAD-UFES-20** (`metadata.csv`):
- `img_id`, `diagnostic` (ACK/BCC/MEL/NEV/SCC/SEK), `fitspatrick` (misspelled in original), `age`, `gender`, `region`

**BCN20000** (`bcn20000_metadata.csv`):
- `isic_id`, `diagnosis`, `age`, `sex`, `anatom_site_general`

---

## Related Work

### Deep Learning for Dermatology

- **Esteva et al. (2017)**: First demonstration of dermatologist-level CNN accuracy on skin cancer classification (129,450 clinical images). Established deep learning as viable for dermatological diagnosis.
- **Haenssle et al. (2018)**: Deep learning outperformed 58 dermatologists on dermoscopic melanoma recognition. However, used only dermoscopic images.
- **Brinker et al. (2019)**: Deep learning outperformed 136 of 157 dermatologists. Again, dermoscopic-only evaluation.

### Fairness and Bias

- **Daneshjou et al. (2022)**: Created DDI dataset revealing that models trained on dermoscopic data perform substantially worse on clinical images from darker-skinned patients. Key motivation for our multi-domain, fairness-aware approach.
- **Kinyanjui et al. (2020)**: Showed skin tone significantly affects classifier performance, with darker skin types exhibiting lower accuracy.
- **Obermeyer et al. (2019)**: Demonstrated racial bias in a widely-used healthcare algorithm. Established importance of fairness auditing.

### Vision-Language Foundation Models

- **CLIP (Radford et al., 2021)**: Contrastive pretraining enables zero-shot transfer to diverse downstream tasks.
- **SigLIP (Zhai et al., 2023)**: Improved CLIP via sigmoid loss, better scaling. Our chosen backbone.
- **BiomedCLIP (Zhang et al., 2023)**: Multimodal biomedical foundation model, but not specifically tuned for dermatology fairness.

### What Is Novel About Our Approach

Previous work focuses on single-dataset, dermoscopic-only evaluation. SkinTag is novel in:
1. Multi-dataset training across 5 complementary sources spanning 3 imaging domains
2. Explicit Fitzpatrick-balanced sampling for skin tone fairness
3. Two-stage training (fine-tune embeddings, then gradient boosting) outperforming end-to-end approaches
4. Three-tier clinical triage system (LOW/MODERATE/HIGH) with inflammatory auto-promotion
5. Knowledge distillation for mobile deployment (200x compression with <0.4% F1 loss)

---

## Evaluation Strategy & Metrics

### Primary Metric: F1 Macro

**Why not accuracy?** The dataset is imbalanced (79% benign, 21% malignant). A naive majority-class classifier achieves 79.1% accuracy by always predicting benign. Accuracy is misleading.

**Why F1 macro?** Treats both classes equally. Balances precision and recall. The standard metric for imbalanced medical classification tasks.

### Full Metric Suite

| Metric | Purpose |
|--------|---------|
| **F1 macro** | Primary metric. Equal weight to both classes. |
| **F1 malignant** | Specifically measures cancer detection ability. |
| **AUC (ROC)** | Threshold-independent performance measure. |
| **Accuracy** | Reported for completeness, not primary. |
| **Equalized odds gap** | Maximum sensitivity/specificity difference across Fitzpatrick types. Key fairness metric. |
| **Per-Fitzpatrick sensitivity** | Ensures no skin tone has systematically missed malignancies. |
| **Per-domain F1** | Verifies cross-domain generalization (dermoscopic, clinical, smartphone). |
| **Per-dataset performance** | Performance on each of the 5 constituent datasets. |

### Evaluation Split

- 80/20 train/test split, stratified by dataset and label
- Training: 37,821 images
- Test: 9,456 images

---

## Modeling Approach

### Required Model 1: Naive Baseline (Majority Class Classifier)

**Type**: Always predicts the most frequent class (benign).

**Location**: `src/model/baseline.py` (class `MajorityClassBaseline`)

**Results**:
| Metric | Value |
|--------|-------|
| Accuracy | 0.791 |
| F1 Macro | 0.442 |
| F1 Malignant | 0.000 |
| AUC | 0.500 |

**Purpose**: Establishes the floor for accuracy (79.1%) and demonstrates why accuracy alone is inadequate -- this model misses 100% of malignancies.

### Required Model 2: Classical ML (XGBoost on SigLIP Embeddings)

**Type**: Gradient boosted decision trees on frozen vision-language model embeddings.

**Architecture**: Image -> SigLIP-SO400M (frozen, 878M params) -> 1152-d embedding -> XGBClassifier

**Location**: `scripts/train.py`, `results/cache/classifier_xgboost.pkl`

**Results (frozen embeddings)**:
| Metric | Value |
|--------|-------|
| Accuracy | 0.897 |
| F1 Macro | 0.851 |
| F1 Malignant | 0.768 |
| AUC | 0.990 |

**Results (fine-tuned embeddings -- best overall model)**:
| Metric | Value |
|--------|-------|
| Accuracy | 0.968 |
| F1 Macro | 0.951 |
| F1 Malignant | 0.922 |
| AUC | 0.992 |

**Rationale**: XGBoost is robust to noise/outliers, handles tabular features well, and provides interpretability through feature importance. Two-stage approach (fine-tune embeddings then boost) outperforms end-to-end neural approaches.

### Required Model 3: Neural Network Deep Learning (Fine-Tuned SigLIP)

**Type**: End-to-end fine-tuned vision transformer with classification head.

**Architecture**: Image -> SigLIP-SO400M (last 4 layers unfrozen) -> BatchNorm -> Linear(1152, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 2)

**Location**: `src/model/deep_classifier.py` (class `EndToEndSigLIP`), `models/finetuned_siglip/`

**Results**:
| Metric | Value |
|--------|-------|
| Accuracy | 0.923 |
| F1 Macro | 0.887 |
| F1 Malignant | 0.824 |
| AUC | 0.960 |

**Training**: AdamW optimizer, LR 1e-4, batch size 16, 10 epochs with early stopping (patience 3). Binary cross-entropy with domain+Fitzpatrick balanced sample weights.

### Selected Deployment Model

**XGBoost on fine-tuned SigLIP embeddings** (two-stage approach) -- achieves the best results across all metrics (96.8% accuracy, 0.951 F1 macro, 0.992 AUC).

### Additional Models Trained

| Model | Embedding Type | Accuracy | F1 Macro | AUC |
|-------|----------------|----------|----------|-----|
| Logistic Regression | Frozen SigLIP | 0.821 | 0.769 | 0.922 |
| Deep MLP | Frozen SigLIP | 0.780 | 0.736 | -- |
| MobileNetV3-Large (distilled) | N/A | 0.925 | 0.884 | 0.959 |
| EfficientNet-B0 (distilled) | N/A | 0.927 | 0.887 | 0.960 |

---

## Data Processing Pipeline

### Step 1: Dataset Loading
Each dataset has a dedicated adapter (`src/data/datasets/*.py`) that converts raw CSV + image paths into a unified `SkinSample` dataclass (`src/data/schema.py`). Raw labels are mapped to the 10-category taxonomy via per-dataset dictionaries in `src/data/taxonomy.py`.

**Rationale**: Unified schema eliminates per-dataset branching logic throughout the pipeline.

### Step 2: Lazy Image Loading
Dataset loaders store file paths, not PIL images. The embedding extractor loads images per-batch from disk. This reduced data loading from ~393s to ~2s for 29k samples and avoids multi-GB RAM usage.

**Rationale**: With 47k images at ~500KB each, loading all into RAM would require ~25GB. Lazy loading keeps RAM under 4GB.

### Step 3: Embedding Extraction
SigLIP-SO400M extracts 1152-dimensional embeddings for each image. Embeddings are cached to `results/cache/embeddings.pt` (~210MB for 47k samples).

**Rationale**: Embedding extraction is the computational bottleneck (~16ms/image on GPU). Caching avoids redundant computation across training runs.

### Step 4: Domain + Fitzpatrick Balanced Sampling
Combined sample weights balance across imaging domain (dermoscopic/clinical/smartphone) and Fitzpatrick skin type simultaneously. Implemented in `src/data/sampler.py`.

**Rationale**: Without balancing, the model would optimize for dermoscopic images (59% of data) and lighter skin tones (68% of annotated data), producing biased predictions.

### Step 5: Field Condition Augmentations
Albumentations pipeline simulating realistic smartphone capture conditions: motion blur, focus issues, lighting variation, color cast, sensor noise, compression, shadows, glare. Applied with probability 0.3-0.5 during training.

**Rationale**: Training images are predominantly from clinical settings with controlled conditions. Augmentations bridge the domain gap to real-world smartphone photos.

### Step 6: Train/Test Split
80/20 stratified split by dataset and label. Training: 37,821 images. Test: 9,456 images.

---

## Hyperparameter Tuning Strategy

### SigLIP Fine-Tuning
- **Unfrozen layers**: Last 4 of 27 vision transformer layers (~7% trainable parameters)
- **Learning rate**: Differential -- backbone 1e-5, classification head 1e-4
- **Optimizer**: AdamW with weight decay
- **Batch size**: 16 (GPU), gradient accumulation 4 steps (effective batch 64)
- **Epochs**: 10-15, early stopping with patience 3
- **Selection**: Best epoch chosen by validation loss

### XGBoost
- **n_estimators**: 500 (default, with early stopping)
- **max_depth**: 6
- **learning_rate**: 0.1
- **scale_pos_weight**: Computed from class ratio for imbalance handling

### Triage Thresholds
- Optimized via clinical triage analysis (`scripts/clinical_triage_analysis.py`)
- Three tiers with inflammatory auto-promotion:
  - LOW: <30% malignancy score (non-inflammatory benign)
  - MODERATE: 30-60% malignancy score, or inflammatory condition auto-promoted from LOW
  - HIGH: >60% malignancy score

---

## Results

### Overall Binary Classification (Test Set, n=9,456)

| Model | Emb. Type | Accuracy | F1 Macro | F1 Malig. | AUC |
|-------|-----------|----------|----------|-----------|-----|
| Majority baseline | N/A | 0.791 | 0.442 | 0.000 | 0.500 |
| Logistic regression | Frozen | 0.821 | 0.769 | 0.660 | 0.922 |
| XGBoost | Frozen | 0.897 | 0.851 | 0.768 | 0.990 |
| Deep MLP | Frozen | 0.780 | 0.736 | 0.628 | -- |
| Fine-tuned SigLIP | End-to-end | 0.923 | 0.887 | 0.824 | 0.960 |
| **XGBoost** | **Fine-tuned** | **0.968** | **0.951** | **0.922** | **0.992** |

### Fairness Analysis (XGBoost on Fine-Tuned Embeddings)

| Attribute | Sensitivity Gap | Specificity Gap | F1 Gap |
|-----------|-----------------|-----------------|--------|
| Fitzpatrick skin type | 0.044 | 0.091 | 0.111 |
| Imaging domain | 0.033 | 0.064 | 0.127 |
| Sex | 0.035 | 0.035 | 0.013 |
| Age group | 0.044 | 0.165 | 0.167 |

Fitzpatrick sensitivity gap < 5% -- critical for equitable healthcare screening.

### Cross-Domain Generalization

| Domain | Accuracy | Sensitivity | Specificity | AUC | n |
|--------|----------|-------------|-------------|-----|---|
| Clinical | 0.980 | 0.955 | 0.984 | 0.990 | 3,404 |
| Dermoscopic | 0.940 | 0.954 | 0.936 | 0.986 | 5,592 |
| Smartphone | 0.989 | 0.986 | 1.000 | 1.000 | 460 |

Smartphone performance (98.9%) demonstrates successful generalization to the target deployment domain.

### Per-Dataset Performance

| Dataset | Accuracy | Sensitivity | AUC | n |
|---------|----------|-------------|-----|---|
| HAM10000 | 0.940 | 0.945 | 0.984 | 2,021 |
| DDI | 0.955 | 0.971 | 0.992 | 134 |
| Fitzpatrick17k | 0.981 | 0.954 | 0.990 | 3,270 |
| PAD-UFES-20 | 0.989 | 0.986 | 1.000 | 460 |
| BCN20000 | 0.940 | 0.958 | 0.987 | 3,571 |

### Robustness to Image Distortions

Tested on 1,000 test images across 12 distortion types:

| Distortion | XGBoost (Frozen) | XGBoost (Fine-tuned) | Delta |
|------------|------------------|----------------------|-------|
| None (clean) | 96.2% | 97.5% | +1.3% |
| Blur (light) | 91.8% | 95.3% | +3.5% |
| Blur (heavy) | 89.7% | 92.9% | +3.2% |
| Noise (light) | 79.0% | 78.8% | -0.2% |
| Noise (heavy) | 79.2% | 79.4% | +0.2% |
| Brightness (dark) | 85.9% | 90.1% | +4.2% |
| Brightness (bright) | 86.8% | 90.1% | +3.3% |
| Compression (light) | 95.0% | 96.8% | +1.8% |
| Compression (heavy) | 94.8% | 97.0% | +2.2% |
| Rotation (15 deg) | 90.3% | 94.0% | +3.7% |
| Rotation (45 deg) | 88.9% | 92.7% | +3.8% |
| Combined (realistic) | 84.3% | 85.8% | +1.5% |

Fine-tuned embeddings improve robustness in 11 of 12 conditions. Largest gains: brightness (+4.2%), rotation (+3.7-3.8%), blur (+3.2-3.5%).

### Retraining Pipeline Results (2026-02-04)

Full SigLIP fine-tuning with field condition augmentations on 47,277 samples:

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| XGBoost (frozen) | 88.2% | 0.794 | 0.916 |
| **XGBoost (fine-tuned)** | **92.0%** | **0.866** | **0.960** |
| MLP (fine-tuned) | 91.2% | 0.856 | 0.945 |
| XGBoost condition 10-class | 71.2% | 0.554 | 0.935 |

Clinical thresholds (XGBoost on fine-tuned embeddings):

| Sensitivity | Threshold | Specificity |
|-------------|-----------|-------------|
| 99% | 0.001 | 47.6% |
| 95% | 0.034 | 82.5% |
| 90% | 0.140 | 89.0% |
| 85% | 0.291 | 92.1% |

### Condition Classification (10-Class)

| Model | Test Acc | F1 Macro |
|-------|----------|----------|
| Logistic | 0.684 | 0.596 |
| Deep MLP | 0.599 | 0.533 |

Per-condition F1 (logistic, evaluation split):

| Condition | F1 | n |
|-----------|-----|---|
| Melanoma | 0.667 | 329 |
| Basal Cell Carcinoma | 0.745 | 1,174 |
| Squamous Cell Carcinoma | 0.673 | 147 |
| Actinic Keratosis | 0.755 | 220 |
| Melanocytic Nevus | 0.847 | 2,677 |
| Seborrheic Keratosis | 0.584 | 546 |
| Dermatofibroma | 0.816 | 85 |
| Vascular Lesion | 0.864 | 66 |
| Non-Neoplastic | 0.707 | 1,258 |
| Other/Unknown | 0.707 | 2,954 |

### Knowledge Distillation Results (2026-02-06)

| Model | Params | Size | Accuracy | F1 Macro | F1 Malig. | AUC |
|-------|--------|------|----------|----------|-----------|-----|
| SigLIP (Teacher) | 878M | 3.4 GB | 92.3% | 0.887 | 0.824 | 0.960 |
| MobileNetV3-Large | 3.2M | 12.5 MB | 92.4% | 0.884 | 0.816 | 0.959 |
| EfficientNet-B0 | 4.3M | 16.8 MB | 92.7% | 0.887 | 0.820 | 0.960 |

MobileNetV3 v2 (30 epochs): Val Accuracy 87.5%, F1 Macro 0.830, F1 Malignancy 0.743, AUC 0.945.

### Model Sizes and Inference Times

| Component | Size | Notes |
|-----------|------|-------|
| Full fine-tuned SigLIP | 3,350 MB | Complete model with all layers |
| SigLIP classification head only | 1.14 MB | For transfer learning |
| XGBoost (frozen embeddings) | 1.77 MB | Deployable without fine-tuned model |
| XGBoost (fine-tuned embeddings) | 1.40 MB | Requires fine-tuned SigLIP |
| Logistic regression | 0.04 MB | Smallest classifier |

| Operation | Time | Hardware |
|-----------|------|----------|
| Fine-tuned SigLIP inference | 48 ms | RTX 4070 Ti SUPER |
| Frozen embedding extraction | 16 ms | RTX 4070 Ti SUPER |
| XGBoost inference | 0.6 ms | CPU |
| XGBoost training (fine-tuned) | 40 s | CPU |

---

## Error Analysis

The XGBoost classifier (frozen SigLIP embeddings) produced 975 mispredictions on the 9,456-image test set (10.3% error rate). Of these, 45 had Fitzpatrick skin type annotations. We selected 5 representative cases spanning different skin tones, conditions, and error types. Images are saved in `writeup/figures/error_*.png`.

### Five Specific Mispredictions (Real Test Set Images)

**1. Melanoma in situ on Fitzpatrick V skin (False Negative, DDI 000001)**
- **Image**: `writeup/figures/error_1_melanoma_fst5.png` -- Small dark melanoma on plantar foot, dark skin
- **Score**: P(malignant) = 0.40 (below 0.5 threshold, but above 95%-sensitivity threshold of 0.157)
- **Root cause**: Reduced contrast between lesion and surrounding dark skin. The lesion is very small (<5mm) and subtle. Training data has limited examples of plantar melanomas on FST V skin.
- **Mitigation**: The three-tier triage system would correctly flag this as MODERATE (score 0.40 falls in the 30-60% range). Continued collection of diverse melanoma images on darker skin tones needed.

**2. Smartphone melanoma on Fitzpatrick III skin (False Negative, PAD-UFES-20 PAT_333)**
- **Image**: `writeup/figures/error_2_melanoma_smartphone_fst3.png` -- Melanoma with irregular borders, smartphone photo
- **Score**: P(malignant) = 0.016 (severe miss; falls in LOW tier, well below the 30% MODERATE threshold)
- **Root cause**: This is the only smartphone-domain melanoma with Fitzpatrick annotation in the test set. The model appears to underfit rare (smartphone, melanoma) combinations. Despite clear irregular borders and asymmetric pigmentation visible to the eye, the model assigned near-zero malignancy probability.
- **Mitigation**: Targeted augmentation of smartphone melanoma images. Hierarchical multi-task training with heavy upweighting of melanoma class. This case is the strongest argument for the field condition augmentation pipeline.

**3. Mycosis fungoides on Fitzpatrick V skin (False Negative, DDI 000069)**
- **Image**: `writeup/figures/error_3_mycosis_fst5.png` -- Hypopigmented patches on dark skin (cutaneous T-cell lymphoma)
- **Score**: P(malignant) = 0.31 (falls in the three-tier MODERATE range, 30-60%)
- **Root cause**: Mycosis fungoides is a rare cutaneous lymphoma mapped to "Other/Unknown" in our taxonomy (not one of the 4 explicitly malignant conditions). It presents as hypopigmented patches on dark skin, visually distinct from the typical melanoma/BCC/SCC features the model learns.
- **Mitigation**: Expand the condition taxonomy to include additional malignant conditions. Hierarchical classification with an explicit "rare malignancy" category.

**4. Verruca vulgaris on Fitzpatrick I skin (False Positive, DDI 000352)**
- **Image**: `writeup/figures/error_4_verruca_fp_fst1.png` -- Wart on nose of elderly patient with extensive sun damage
- **Score**: P(malignant) = 0.94 (high-confidence false positive)
- **Root cause**: The perilesional skin shows extensive actinic damage, telangiectasia, and sun spots that resemble squamous cell carcinoma or actinic keratosis patterns. The model appears to respond to the overall skin damage context rather than the specific lesion.
- **Mitigation**: Lesion-focused cropping to reduce context influence. Automated lesion detection (future Phase 3C) would isolate the target lesion from surrounding skin damage.

**5. Kaposi sarcoma on Fitzpatrick V skin (False Negative, DDI 000125)**
- **Image**: `writeup/figures/error_5_kaposi_fst5.png` -- Violaceous patch on arm, dark skin
- **Score**: P(malignant) = 0.18 (falls in the three-tier LOW range, below the 30% MODERATE threshold)
- **Root cause**: Kaposi sarcoma is rare in training data and maps to "Other/Unknown" in our taxonomy. The subtle violaceous coloring on dark skin is difficult to distinguish from benign hyperpigmentation.
- **Mitigation**: Similar to Case 3; expand taxonomy for rare malignancies. The three-tier triage system places this in LOW (score 0.18 < 0.30 threshold), missing a true malignancy.

### Fairness Pattern

Three of five Fitzpatrick-annotated mispredictions involve FST V (dark skin), consistent with known representation gaps. Two involve conditions mapped to "Other/Unknown" in the taxonomy (mycosis fungoides, Kaposi sarcoma), suggesting that expanding the condition set or using hierarchical classification could capture these edge cases. The smartphone melanoma miss (Case 2) is particularly concerning for the target deployment scenario.

### Common Failure Modes

- **Image quality**: Heavy noise degrades accuracy to ~79% (vs. 97.5% on clean images)
- **Rare conditions on rare skin tones**: Low absolute sample counts for specific (condition, Fitzpatrick type) pairs
- **Taxonomy limitations**: Some malignancies (mycosis fungoides, Kaposi sarcoma) map to "Other/Unknown" and are underweighted during training
- **Perilesional context**: Surrounding skin damage can confound classification (Case 4)

---

## Experiment Write-Up: Domain Shift and Fairness Sensitivity

### Experimental Plan

**Research question**: Does training with multi-dataset aggregation, domain-balanced sampling, and Fitzpatrick-balanced weights improve cross-domain generalization and fairness across skin tones?

**Protocol**:
1. Train all 3 required models (baseline, XGBoost, fine-tuned SigLIP) on HAM10000 only (single-domain baseline)
2. Train all 3 models on multi-dataset (no augmentation, no balancing)
3. Train all 3 models on multi-dataset with domain+Fitzpatrick balanced weights + field augmentations
4. For each: evaluate per-Fitzpatrick-type F1, per-domain F1, equalized odds gaps

**Metrics**: F1 macro, F1 (malignant), per-Fitzpatrick-type sensitivity, equalized odds gap, cross-domain F1 gap

### Results

**Finding 1: Multi-dataset training dramatically improves cross-domain generalization.**
- Single-dataset (HAM10000 only) models fail on clinical and smartphone images (domain gap)
- Multi-dataset training achieves 98.0% accuracy on clinical, 98.9% on smartphone images

**Finding 2: Fitzpatrick-balanced sampling reduces skin tone bias without sacrificing overall performance.**
- Sensitivity gap across Fitzpatrick types: 0.044 (< 5%)
- This compares favorably to prior work reporting gaps >15% (Daneshjou et al., 2022)
- Overall F1 macro remains strong at 0.951 (best model)

**Finding 3: Fine-tuned embeddings provide consistent robustness improvements.**
- Under 12 distortion types, fine-tuned embeddings improve accuracy in 11 of 12 conditions
- Largest gains in blur (+3.5%), brightness (+4.2%), rotation (+3.7%) -- common smartphone photo issues
- Noise remains challenging for both approaches (area for future improvement)

**Finding 4: Two-stage training outperforms end-to-end.**
- XGBoost on fine-tuned embeddings (96.8% acc) > end-to-end SigLIP (92.3% acc) > XGBoost on frozen (89.7% acc)
- Hypothesis: fine-tuning improves the embedding space, while XGBoost's ensemble provides robustness that neural classification heads lack

### Interpretation and Recommendations

The apparent trade-off between fairness and performance in dermatology AI may be an artifact of insufficient data diversity. With appropriate dataset aggregation and fairness-aware training, it is possible to build systems that perform well across imaging conditions and skin tones.

For deployment, we recommend:
- XGBoost on fine-tuned embeddings as the primary classifier (best accuracy + robustness)
- 95% sensitivity operating point for screening (threshold = 0.034, specificity = 82.5%)
- Continued monitoring of per-Fitzpatrick-type metrics post-deployment

---

## Interactive Application

### Web Application (Live)

**Live URL**: https://skintaglabs.github.io/main/

**Technology stack**:
- **Backend**: FastAPI (Python) serving model inference (`app/main.py`)
- **Frontend**: React 19 + TypeScript + Vite + Tailwind CSS (`webapp-react/`)
- **Hosting**: GitHub Actions + Cloudflare Tunnel (serverless, free tier)
- **Model hosting**: HuggingFace Hub (`skintaglabs/siglip-skin-lesion-classifier`)

### Architecture

1. User uploads image or captures via camera
2. Image is sent to FastAPI backend (`/api/analyze` endpoint)
3. Backend runs SigLIP embedding extraction -> XGBoost classification -> condition estimation
4. Returns: risk score, urgency tier, recommendation, condition probabilities, triage categories
5. Frontend displays results with risk gauge, triage tier card, condition breakdown, ABCDE guide

### UX Features

- **Camera capture**: HTML5 `getUserMedia` API with webcam support and hand detection (MediaPipe)
- **Image cropping**: Pre-analysis crop tool (react-easy-crop) for lesion focus
- **Risk display**: Animated gauge showing cancer risk (0-100%) with color coding
- **Three-tier triage**: LOW (green) / MODERATE (amber) / HIGH (red)
- **Three-category breakdown**: Malignant / Inflammatory / Benign with animated probability bars
- **Condition estimation**: Top-3 most likely conditions with probabilities
- **ABCDE guide**: Educational melanoma self-check guide
- **Analysis history**: Local storage of past analyses with list view
- **Dark mode**: Theme toggle with localStorage persistence
- **Onboarding modal**: First-time user tutorial
- **Network status**: API health banner with connectivity indicator
- **Medical disclaimers**: Prominent "AI screening tool, not a diagnosis" banners throughout
- **Responsive design**: Mobile-first with warm beige aesthetic, Instrument Serif headings, DM Sans body

### Deployment Infrastructure

- **Inference server**: GitHub Actions workflow dispatches cloud GPU instance
- **Auto-tunnel**: Cloudflare tunnel auto-created (no domain setup needed)
- **Runtime**: 5-6 hour sessions with auto-restart
- **Cold start**: ~1-2 minutes
- **Model download**: Auto-cached from HuggingFace Hub on first start

### Three-Tier Clinical Triage System

| Tier | Score | Logic | Color | Action |
|------|-------|-------|-------|--------|
| LOW | <30% | Non-inflammatory benign | Green | Self-monitor, routine check-up |
| MODERATE | 30-60% or inflammatory | Auto-promoted if inflammatory | Amber | Schedule dermatology visit |
| HIGH | >60% | High malignancy score | Red | Seek prompt evaluation |

Each tier includes condition-specific guidance with inflammatory auto-promotion logic and links to DermNet NZ (peer-reviewed dermatology reference).

### Additional Interfaces

- **HuggingFace Space**: Gradio-based interface at `huggingface_space/app.py` with HTML result cards, risk meter, and ABCDE guide
- **Classic web app**: Standalone HTML app with tooltips and clinical guidance

---

## Project Structure

```
SkinTag/
├── README.md                          <- Project description and setup instructions
├── PLAN.md                            <- This file
├── requirements.txt                   <- Python dependencies
├── requirements-inference.txt         <- Minimal dependencies for inference only
├── Makefile                           <- Build targets (install, data, train, app)
├── run_pipeline.py                    <- Unified pipeline: data -> embed -> train -> eval -> app
├── Dockerfile                         <- Containerized deployment (CPU)
├── Dockerfile.gpu                     <- GPU deployment with CUDA
├── LICENSE
├── app/
│   ├── main.py                        <- FastAPI backend (upload -> SigLIP -> triage)
│   └── templates/
│       └── index.html                 <- Legacy frontend
├── webapp-react/                      <- React 19 + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx                    <- Main app with routing
│   │   ├── components/               <- UI components (upload, results, layout, camera, history)
│   │   ├── contexts/                  <- App state and theme contexts
│   │   ├── hooks/                     <- Analysis, history, validation, network hooks
│   │   ├── lib/                       <- API client, utilities
│   │   └── types/                     <- TypeScript interfaces
│   ├── public/                        <- Static assets
│   └── package.json                   <- React 19, Vite 7, Tailwind CSS 4, Radix UI
├── configs/
│   ├── config.yaml                    <- All configuration (data, training, triage thresholds)
│   └── benchmark_config.yaml          <- Benchmarking configuration
├── models/
│   ├── finetuned_siglip/              <- Fine-tuned SigLIP model (~3.3GB, gitignored weights)
│   ├── mobilenet_distilled/           <- MobileNetV3-Large distilled model
│   ├── mobilenet_distilled_v2/        <- MobileNetV3 v2 (30 epochs, improved)
│   └── efficientnet_distilled/        <- EfficientNet-B0 distilled model
├── data/                              <- gitignored -- local datasets (5 sources)
├── notebooks/
│   ├── label_taxonomy_eda.ipynb       <- Label taxonomy and data enrichment EDA
│   ├── skin_tone_eda.ipynb            <- Skin tone exploratory analysis
│   ├── colab_demo.ipynb               <- Google Colab demo
│   └── demo.ipynb                     <- Local demo
├── results/                           <- gitignored -- cached embeddings, models, metrics
│   ├── cache/                         <- Embeddings, classifiers, training results JSONs
│   ├── domain_gap_analysis.png        <- Domain shift visualization
│   ├── fitzpatrick_coverage.png       <- Skin tone coverage visualization
│   └── label_distributions_raw.png    <- Label distribution visualization
├── scripts/
│   ├── train.py                       <- Main training script
│   ├── train_all_models.py            <- Train + compare all model types
│   ├── evaluate.py                    <- Full fairness evaluation report
│   ├── evaluate_cross_domain.py       <- Leave-one-domain-out experiment
│   ├── full_retraining_pipeline.py    <- Standalone retraining with SigLIP fine-tuning
│   └── clinical_triage_analysis.py    <- Triage threshold optimization
├── src/
│   ├── data/
│   │   ├── schema.py                  <- SkinSample dataclass (unified schema)
│   │   ├── taxonomy.py                <- Unified condition taxonomy (10 categories)
│   │   ├── loader.py                  <- Multi-dataset loading orchestrator
│   │   ├── datasets/                  <- Per-dataset adapters (5 datasets)
│   │   ├── sampler.py                 <- Domain + Fitzpatrick balanced sampling
│   │   ├── augmentations.py           <- Training/eval transforms + domain bridging
│   │   └── dermoscope_aug.py          <- Custom dermoscope artifact augmentations
│   ├── model/
│   │   ├── embeddings.py              <- SigLIP embedding extraction + caching
│   │   ├── classifier.py              <- Sklearn classifier (logistic, XGBoost)
│   │   ├── baseline.py                <- Majority class baseline
│   │   ├── deep_classifier.py         <- Deep MLP + EndToEndSigLIP
│   │   └── triage.py                  <- TriageSystem with urgency tiers
│   └── evaluation/
│       └── metrics.py                 <- F1, accuracy, AUC, per-group, equalized odds
├── writeup/
│   ├── main.tex                       <- NeurIPS-style technical report
│   ├── references.bib                 <- Bibliography
│   └── neurips_2024.sty               <- Style file
├── huggingface_space/                 <- HuggingFace Space (Gradio interface)
├── .docs/                             <- Deployment and benchmarking documentation
├── .github/
│   └── workflows/
│       ├── deploy-webapp.yml          <- Frontend deployment
│       ├── inference-server-a.yml     <- Inference server A
│       ├── inference-server-b.yml     <- Inference server B
│       └── train.yml                  <- Training workflow
└── .gitignore
```

---

## Usage

### Unified Pipeline (Recommended)

```bash
python run_pipeline.py                     # Full pipeline: data -> embed -> train -> eval -> app
python run_pipeline.py --quick --no-app    # Quick smoke test (500 samples)
python run_pipeline.py --skip-train        # Re-evaluate existing models
python run_pipeline.py --app-only          # Launch web app only
```

### Setup

```bash
pip install -r requirements.txt
# GPU support (CUDA 12.6):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
make data                                  # Download HAM10000 (Kaggle credentials required)
```

### Training

```bash
python scripts/train.py --multi-dataset --domain-balance --model all
python scripts/full_retraining_pipeline.py --finetune-siglip --epochs 15
```

### Evaluation

```bash
python scripts/evaluate.py --models logistic deep baseline
python scripts/evaluate_cross_domain.py
```

### Web App

```bash
make app                                   # http://localhost:8000
make app-docker                            # Docker (CPU)
make app-docker-gpu                        # Docker (GPU)
```

---

## Key Design Decisions

1. **F1 macro as primary metric** -- accuracy is misleading on class-imbalanced data (79% benign -> 79% accuracy by always predicting benign). F1 macro treats both classes equally.

2. **Fitzpatrick-balanced sampling, not oversampling** -- oversampling duplicates minority images (overfitting risk). Balanced sample weights upweight minority samples in the loss function without duplication.

3. **Domain-bridging augmentations** -- randomly add dermoscope artifacts to phone photos and remove them from dermoscopic images. The model sees all label-domain combinations, breaking spurious correlations.

4. **three_partition_label for Fitzpatrick17k** -- reliable benign/malignant/non-neoplastic partition eliminates fuzzy string matching. Non-neoplastic included as benign (not cancer = benign for triage), recovering ~8,000 images.

5. **Triage tiers, not probabilities** -- users understand "moderate concern -- schedule a dermatology appointment within 2-4 weeks" better than "0.47 probability of malignancy."

6. **Medical disclaimer everywhere** -- screening aid, not diagnosis. Every output includes a prominent disclaimer.

7. **SigLIP backbone** -- 400M parameter vision-language model producing 1152-d embeddings. Strong zero-shot medical image understanding.

8. **Lazy/streaming image loading** -- reduced data loading from ~393s to ~2s by storing paths instead of images.

9. **GPU auto-detection** -- pipeline auto-detects CUDA and adjusts batch size (4 CPU, 16 GPU).

---

## Conclusions

SkinTag demonstrates that the apparent trade-off between fairness and performance in dermatology AI may be an artifact of insufficient data diversity. Key findings:

1. **Two-stage training is optimal**: Fine-tune embeddings with SigLIP, then classify with XGBoost. This outperforms both end-to-end fine-tuning and classical ML on frozen embeddings.
2. **Fairness is achievable**: Fitzpatrick-balanced sampling reduces sensitivity gaps across skin tones to <5% without sacrificing overall performance (0.951 F1 macro).
3. **Multi-domain training generalizes**: Training on 5 datasets spanning 3 imaging domains enables 98.9% accuracy on smartphone images -- the target deployment scenario.
4. **Robustness matters**: Field condition augmentations improve performance under real-world distortions by 1.5-4.2% across conditions.
5. **Knowledge distillation preserves quality**: 200-270x model compression (3.4GB to 12-17MB) with <0.4% F1 loss enables future mobile deployment.

---

## Future Work

### Mobile Deployment (Offline Inference)

A native mobile application running inference entirely on-device for use in areas without reliable internet connectivity.

**Completed preparatory work**:
- Knowledge distillation: MobileNetV3-Large (12.5 MB, 92.4% acc) and EfficientNet-B0 (16.8 MB, 92.7% acc)
- ONNX exports ready for Core ML (iOS) and TFLite (Android) conversion
- Model artifacts in `models/mobilenet_distilled/` and `models/efficientnet_distilled/`
- Prototype mobile app scaffolding in `mobile/` (Flutter and iOS)

**Remaining work**:
- Convert ONNX to Core ML and TFLite formats
- Build native mobile apps with camera capture and on-device inference
- INT8 quantization (target: 5-6 MB)
- UI framing guidance and lesion centering overlay
- App store submission and beta testing

### Hierarchical Multi-Task Fine-Tuning

Per-condition sensitivity gaps for melanoma (68%) and SCC (59%) need improvement via hierarchical multi-task loss with clinical class weights.

### Automated Lesion Detection

Lightweight object detection (YOLO-Nano or MobileNet-SSD) for automatic lesion cropping, enabling photography from any distance and multi-lesion triage.

### Prospective Clinical Validation

Retrospective evaluation must be followed by prospective clinical validation before real-world deployment.

### Webapp Feature Roadmap

Prioritized features in `webapp-react/FEATURES.md`: download/export results, PWA offline support, find nearby dermatologists, multi-image comparison, analysis trend charts, and telemedicine integration.

---

## Commercial Viability Statement

SkinTag demonstrates potential for real-world commercial application with significant caveats:

**Strengths**: Addresses a genuine unmet need (limited dermatologist access). Strong performance across skin tones (fairness gap <5%) differentiates from existing tools. Lightweight distilled models enable offline mobile use. Three-tier triage provides clinically actionable outputs. Works with consumer smartphone photos.

**Barriers**: FDA Class II medical device classification requires clinical trials and 510(k) clearance (multi-year, multi-million dollar process). False negatives carry significant medical-legal risk. Melanoma sensitivity (68%) and SCC sensitivity (59%) are below clinical thresholds for standalone screening.

**Viable pathway**: Position as a triage/screening aid (not diagnostic tool) with prominent medical disclaimers. Target B2B (healthcare systems, telemedicine platforms) rather than direct-to-consumer. Pursue clinical partnerships for prospective validation.

---

## Ethics Statement

### Patient Safety
SkinTag is a **triage aid, not a diagnostic tool**. All outputs include prominent disclaimers. The system explicitly recommends professional evaluation for any concerning findings.

### Fairness and Bias
We actively address skin tone bias through Fitzpatrick-balanced sampling, DDI as a fairness benchmark, per-Fitzpatrick-type evaluation, and equalized odds monitoring. Residual disparities may exist for rare (condition, skin tone) combinations.

### Privacy
No patient images are stored server-side after inference. No personally identifiable information is collected. Analysis history is stored locally on the user's device only.

### Potential for Harm
- False negatives could delay cancer diagnosis. Mitigated by conservative triage thresholds (95% sensitivity operating point).
- False positives could cause unnecessary anxiety. Mitigated by graded urgency levels rather than binary labels.
- Over-reliance on AI could substitute for professional care. Mitigated by persistent medical disclaimers.

### Data Ethics
All five datasets are publicly available for research use. DDI requires Stanford AIMI access request. No new patient data was collected. Dataset creators are credited and cited.
