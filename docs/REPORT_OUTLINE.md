# SkinTag: Report Outline

Target format: **NeurIPS/ICML-style technical report**

---

## 1. Abstract (~150 words)

- Problem: Skin cancer is the most common cancer; early detection saves lives but dermatologist access is inequitable
- Approach: Multi-dataset, fairness-aware triage system using SigLIP vision embeddings + 4 classifier heads
- Key results: Binary F1=0.878 (deep MLP), AUC=0.980; Condition F1=0.736; Fitzpatrick fairness gap <0.10
- Novelty: 5-dataset unification with explicit cross-domain and skin tone fairness evaluation; dual-target (binary triage + 10-class condition estimation)

---

## 2. Introduction (~1 page)

- Skin cancer incidence statistics (cite WHO/ACS)
- Wait times for dermatology appointments (cite access disparity studies)
- AI screening tools as a triage layer (not diagnosis)
- Problem: Existing models train on single datasets (HAM10000 or ISIC), lack diversity in skin tones and imaging domains
- Contribution summary:
  1. Unified 5-dataset pipeline (47,277 images, 3 imaging domains)
  2. Dual-target classification (binary triage + 10-class condition estimation)
  3. Explicit Fitzpatrick-stratified fairness evaluation with equalized odds
  4. XGBoost + deep MLP + end-to-end fine-tuning comparison on same data
  5. Deployed web app with mobile camera support

---

## 3. Related Work (~1 page)

### 3.1 Skin Lesion Classification
- ISIC challenges (2016–2024): CNN-based approaches, EfficientNet, Vision Transformers
- HAM10000 benchmarks (Tschandl et al. 2018)
- Transfer learning from ImageNet → dermoscopy

### 3.2 Foundation Models for Medical Imaging
- SigLIP (Zhai et al. 2023): Sigmoid loss for contrastive pre-training
- CLIP-based medical imaging (BiomedCLIP, PubMedCLIP)
- Linear probing vs fine-tuning trade-offs

### 3.3 Fairness in Dermatology AI
- Fitzpatrick17k dataset (Groh et al. 2021) — skin tone annotation
- DDI dataset (Daneshjou et al. 2022) — diverse dermatology images
- Documented bias: models underperform on darker skin tones (cite Adamson & Smith 2018)

### 3.4 Multi-Domain Learning
- Domain adaptation in medical imaging
- Cross-domain evaluation: dermoscopic → clinical → smartphone gap

---

## 4. Data Sources (~1 page)

### 4.1 Datasets

| Dataset | N | Domain | Skin Tones | Source |
|---------|---|--------|------------|--------|
| HAM10000 | 10,015 | Dermoscopic | Limited | ISIC |
| BCN20000 | 17,790 | Dermoscopic | Limited | Hospital Clinic Barcelona |
| Fitzpatrick17k | 16,577 | Clinical | FST I–VI | Atlas images |
| PAD-UFES-20 | 2,298 | Smartphone | Mixed | Brazilian clinic |
| DDI | 597 | Clinical | FST I–VI (balanced) | Stanford |

### 4.2 Unified Condition Taxonomy
- 10-category mapping across all datasets
- Table: Condition → binary (benign/malignant) mapping
- Keyword-based matching for free-text labels (DDI, Fitzpatrick17k)

### 4.3 Label Quality & Limitations
- HAM10000/BCN20000: histopathology-confirmed
- DDI: biopsy-proven
- Fitzpatrick17k: atlas labels (noisy, no biopsy)
- PAD-UFES: clinical diagnosis
- Acknowledge: label noise affects ceiling performance

---

## 5. Evaluation Strategy & Metrics (~0.5 page)

### 5.1 Primary Metric: F1 Macro
- Justification: class imbalance (79% benign, 21% malignant)
- F1 macro weights both classes equally — prevents accuracy inflation from majority class

### 5.2 Secondary Metrics
- **AUC-ROC**: Threshold-independent discrimination
- **Balanced accuracy**: Same as sensitivity-specificity average
- **Sensitivity (recall)**: Clinically critical — missed malignancy is dangerous
- **Specificity**: Reduces unnecessary referrals

### 5.3 Fairness Metrics
- **Accuracy gap**: Max accuracy difference across Fitzpatrick types
- **Equalized odds gap**: Max sensitivity/specificity difference across groups
- **Per-domain accuracy**: Dermoscopic vs clinical vs smartphone

---

## 6. Modeling Approach (~2 pages)

### 6.1 Architecture Overview
- Diagram: Image → SigLIP (frozen) → 1152-d embedding → Classifier heads
- SigLIP-SO400M: 878M params, 384×384 input, sigmoid loss pre-training

### 6.2 Data Processing Pipeline
1. **Loading**: Per-dataset loaders with unified SkinSample schema
2. **Taxonomy mapping**: Raw labels → 10 conditions → binary
3. **Stratified split**: (label × domain) stratification to prevent leakage
4. **Balanced sampling**: Domain + Fitzpatrick combined weights
5. **Embedding extraction**: Batch extraction, fp16 on GPU, cached to disk

### 6.3 Models Evaluated

#### Naive Baseline
- **Majority class**: Always predicts benign (expected: acc=79%, F1=0.44)
- Purpose: Establishes floor — any model must beat this

#### Classical ML: Logistic Regression
- StandardScaler + LogisticRegression(max_iter=1000)
- Domain-balanced sample weights
- Fast (<1s training), interpretable coefficients

#### Classical ML: XGBoost Gradient Boosting
- 500 trees, max_depth=6, learning_rate=0.05
- Subsample=0.8, colsample_bytree=0.8 (regularization)
- Histogram-based splitting for speed on 1152-d embeddings
- Handles non-linear feature interactions that logistic regression misses

#### Deep Learning: MLP Head
- 1152 → 256 (BatchNorm, ReLU, Dropout 0.3) → 2
- AdamW, cosine LR schedule, early stopping (patience=8)
- Class-weighted cross-entropy loss

#### Deep Learning: End-to-End Fine-Tuning
- Unfreezes last 4 SigLIP vision transformer layers
- Differential LR: head=1e-3, backbone=1e-5
- 10 epochs, batch_size=8, early stopping
- Adapts SigLIP features from web images → dermoscopy

### 6.4 Hyperparameter Tuning Strategy
- XGBoost: grid search on max_depth {4,6,8}, n_estimators {300,500,800}, learning_rate {0.01,0.05,0.1}
- Deep MLP: hidden_dim {128,256,512}, dropout {0.2,0.3,0.5}, lr {5e-4,1e-3,5e-3}
- End-to-end: unfreeze_layers {2,4,6}, lr_backbone {1e-6,1e-5,5e-5}
- Selection metric: F1 macro on held-out validation set

---

## 7. Results (~2 pages)

### 7.1 Model Comparison Table

| Model | Test Acc | Balanced Acc | F1 Macro | F1 Malignant | AUC |
|-------|----------|-------------|----------|-------------|-----|
| Majority baseline | 0.791 | 0.500 | 0.442 | 0.000 | 0.500 |
| Logistic regression | 0.840 | — | 0.792 | — | 0.922 |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| Deep MLP | 0.908 | — | 0.878 | — | 0.980 |
| Fine-tuned SigLIP | TBD | TBD | TBD | TBD | TBD |

### 7.2 Condition Classification (10-class)

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| Logistic | 0.684 | 0.596 |
| Deep MLP | TBD | TBD |

### 7.3 Fairness Results
- Per-Fitzpatrick accuracy table
- Equalized odds gaps (sensitivity, specificity)
- Discussion: which skin tones underperform and why

### 7.4 Confusion Matrix
- Binary: 2×2 matrix for best model
- Condition: 10×10 heatmap showing common confusions

### 7.5 ROC Curves
- Overlay all models on same plot
- Mark operating points for low/moderate/high triage thresholds

---

## 8. Error Analysis (~1 page)

### 8.1 Methodology
- Identify highest-confidence mispredictions from best model
- Categorize by: domain, skin tone, condition type

### 8.2 Five Specific Mispredictions
(Fill from `scripts/error_analysis.py` output)

1. **Error #1**: [image, true label, predicted, confidence, root cause]
2. **Error #2**: ...
3. **Error #3**: ...
4. **Error #4**: ...
5. **Error #5**: ...

### 8.3 Root Cause Patterns
- Domain shift (smartphone → dermoscopic training bias)
- Visually ambiguous conditions (melanoma vs seborrheic keratosis)
- Under-represented skin tones (FST V–VI)
- Label noise in atlas-sourced datasets

### 8.4 Mitigation Strategies
1. End-to-end fine-tuning to learn dermoscopy-specific texture features
2. Increased Fitzpatrick V–VI sample weighting
3. Probability calibration (temperature scaling)
4. Lower triage threshold to prioritize sensitivity over specificity
5. Cross-domain augmentation (domain bridging)

---

## 9. Experiment: Cross-Domain Robustness (~1.5 pages)

### 9.1 Experimental Plan
- **Question**: How well does the model generalize across imaging domains?
- **Method**: Leave-one-domain-out evaluation
  - Train on 2 domains, test on held-out domain
  - Compare: {dermoscopic, clinical, smartphone} as held-out
  - Compare: balanced vs unbalanced training weights
  - Models: baseline, logistic, XGBoost, deep MLP

### 9.2 Results Table

| Model | Domain Balanced | Held-Out Domain | Accuracy | F1 Macro | Gap |
|-------|----------------|-----------------|----------|----------|-----|
| ... | ... | ... | ... | ... | ... |

### 9.3 Interpretation
- Which domain transfers worst? (Expect: smartphone, least similar to dermoscopic)
- Does domain balancing help? (Compare balanced vs unbalanced)
- Which model is most robust to domain shift?

### 9.4 Recommendations
- Domain-balanced sampling reduces gap by X%
- Fine-tuning further reduces domain gap
- Smartphone domain remains challenging — dedicated smartphone data collection recommended

---

## 10. Conclusions (~0.5 page)

- Best model: [Deep MLP / Fine-tuned SigLIP] achieves F1=X on 47K multi-domain images
- Multi-dataset approach reduces domain bias compared to single-dataset training
- Fairness gap under X% across Fitzpatrick types with balanced sampling
- XGBoost closes much of the logistic→deep gap as a strong classical baseline
- Dual-target (triage + condition) provides more actionable output than binary alone

---

## 11. Future Work (~0.5 page)

- **More data**: ISIC 2024 challenge data, PH² dataset, proprietary clinical data
- **Better backbone**: BiomedCLIP or domain-specific medical vision encoders
- **Uncertainty quantification**: Monte Carlo dropout or ensemble disagreement
- **Multi-modal**: Incorporate patient metadata (age, sex, location) as auxiliary features
- **Segmentation**: Lesion boundary segmentation for ABCDE feature extraction
- **Longitudinal**: Track lesion evolution over time from sequential photos

---

## 12. Commercial Viability Statement (~0.5 page)

- **Viable as screening aid**: NOT a diagnostic device, but triages effectively
- **Regulatory**: Would require FDA 510(k) / CE marking for clinical use
- **Market**: Teledermatology market projected $X billion by 2030
- **Deployment**: Low-cost (CPU inference on free-tier cloud), mobile-friendly
- **Limitations for commercial use**: Needs clinical validation study, IRB-approved trials
- **Competitive landscape**: SkinVision, MoleScope, DermAssist — SkinTag differentiates on fairness-awareness and multi-domain robustness

---

## 13. Ethics Statement (~0.5 page)

- **Bias**: Training data skews toward lighter skin tones; we explicitly measure and mitigate this
- **Intended use**: Screening aid to encourage dermatology visits, NOT to replace doctors
- **Risk of harm**: False negatives could delay treatment; false positives cause anxiety
- **Disclaimers**: Every prediction includes mandatory medical disclaimer
- **Data ethics**: All datasets are publicly available, IRB-approved, and anonymized
- **Dual use**: Could theoretically be used by unqualified providers to avoid referrals — we discourage this via prominent disclaimers

---

## References

(To be populated — key citations)

- Tschandl et al. (2018) — HAM10000 dataset
- Groh et al. (2021) — Fitzpatrick17k
- Daneshjou et al. (2022) — DDI dataset
- Combalia et al. (2019) — BCN20000
- Pacheco et al. (2020) — PAD-UFES-20
- Zhai et al. (2023) — SigLIP
- Radford et al. (2021) — CLIP
- Esteva et al. (2017) — Dermatologist-level skin cancer classification
- Adamson & Smith (2018) — Machine learning and health care disparities in dermatology

---

## Appendices

- A: Full per-condition classification results
- B: Per-Fitzpatrick fairness breakdown tables
- C: Cross-domain experiment full results
- D: Hyperparameter search results
- E: App screenshots (desktop + mobile)
