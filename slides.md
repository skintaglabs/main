# SkinTag
## Robust Skin Lesion Classification

---

# Problem

Medical images captured under inconsistent conditions:
- Different cameras and lighting
- Compression from uploads/telemedicine
- Sensor noise from low-light capture

Models trained on clean clinical images fail on real-world images.

---

# Approach

| Component | Choice |
|-----------|--------|
| Pre-trained Model | MedSigLIP (400M params) |
| Transfer Learning | Extract embeddings → Logistic Regression |
| Dataset | HAM10000 (10,015 skin lesion images) |

---

# Data Augmentation

Simulate real-world imaging variations:

| Augmentation | Real-World Scenario |
|--------------|---------------------|
| Lighting | Different exam room lighting |
| Noise | Low-light photos, sensor noise |
| Compression | Telemedicine, image uploads |

[Insert augmentation visualization image]

---

# Results

**Classifier Performance**
- Train: 100%
- Test: 86%

**Robustness Across Conditions**

| Condition | Accuracy |
|-----------|----------|
| Original | 74% |
| Lighting | 77% |
| Noise | 80% |
| Compression | 72% |

Model maintains performance under degraded conditions.

---

# Takeaway

MedSigLIP embeddings + targeted augmentations produce a classifier robust to real-world imaging variations.

**Code**: github.com/MedGemma540/SkinTag
