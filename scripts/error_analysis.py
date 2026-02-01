"""Error analysis — identify and characterize the worst mispredictions.

Rubric requirement: identify 5 specific mispredictions, explain root causes,
and propose concrete mitigation strategies.

Outputs:
  - Console report with 5 worst mispredictions
  - results/cache/error_analysis.json with full structured results
  - results/cache/error_analysis_images/ with copies of misclassified images

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --model deep --top 10
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pickle
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.loader import get_demographic_groups
from src.data.taxonomy import CONDITION_NAMES, Condition


def _classify_error_type(true_label, pred_label, confidence, metadata_row):
    """Heuristic root-cause classification for a misprediction."""
    is_fn = true_label == 1 and pred_label == 0  # missed malignant
    is_fp = true_label == 0 and pred_label == 1  # false alarm

    causes = []

    # High confidence wrong = model has learned wrong feature
    if confidence > 0.85:
        causes.append("high-confidence error (model overfit on spurious feature)")

    # Domain-related
    domain = str(metadata_row.get("domain", "unknown"))
    if domain == "smartphone":
        causes.append("smartphone image quality (lighting, blur, artifacts)")
    elif domain == "clinical":
        causes.append("clinical photo (different framing vs dermoscopic training data)")

    # Condition-related
    condition = metadata_row.get("condition_label", None)
    if condition is not None and not np.isnan(float(condition)):
        cond = Condition(int(condition))
        cond_name = CONDITION_NAMES.get(cond, "Unknown")
        # Conditions known to be visually ambiguous
        ambiguous_pairs = {
            Condition.MELANOMA: "melanoma mimics seborrheic keratosis / nevus",
            Condition.SEBORRHEIC_KERATOSIS: "seborrheic keratosis mimics melanoma",
            Condition.ACTINIC_KERATOSIS: "actinic keratosis (pre-cancerous) looks benign early",
            Condition.DERMATOFIBROMA: "dermatofibroma can appear pigmented like melanoma",
        }
        if cond in ambiguous_pairs:
            causes.append(f"visually ambiguous condition ({ambiguous_pairs[cond]})")

    # Fitzpatrick-related
    fitz = metadata_row.get("fitzpatrick", None)
    if fitz is not None:
        try:
            fitz_val = int(float(fitz))
            if fitz_val >= 5:
                causes.append(f"darker skin tone (Fitzpatrick {fitz_val}) — under-represented in training")
        except (ValueError, TypeError):
            pass

    # False negative is clinically worse
    if is_fn:
        causes.append("FALSE NEGATIVE — missed malignancy (clinically dangerous)")
    elif is_fp:
        causes.append("false positive — unnecessary referral but not dangerous")

    if not causes:
        causes.append("no obvious single root cause identified")

    return causes


def _suggest_mitigations(error_causes_all):
    """Aggregate mitigations from all error causes."""
    mitigations = []
    cause_text = " ".join(str(c) for causes in error_causes_all for c in causes)

    if "smartphone" in cause_text:
        mitigations.append(
            "Augment training with more smartphone images or apply domain bridging "
            "(add/remove dermoscope artifacts) to reduce domain gap."
        )
    if "darker skin" in cause_text or "Fitzpatrick" in cause_text:
        mitigations.append(
            "Increase Fitzpatrick V–VI sample weight or collect more diverse skin tone "
            "training data to reduce skin tone bias."
        )
    if "high-confidence" in cause_text:
        mitigations.append(
            "Apply temperature scaling or Platt calibration to improve probability "
            "calibration and reduce overconfident wrong predictions."
        )
    if "visually ambiguous" in cause_text:
        mitigations.append(
            "Fine-tune SigLIP end-to-end on dermoscopy-specific features to learn "
            "subtle texture differences between visually similar conditions."
        )
    if "FALSE NEGATIVE" in cause_text:
        mitigations.append(
            "Lower the triage threshold for 'high concern' to increase sensitivity "
            "at the cost of more false positives (clinically safer trade-off)."
        )
    if "clinical photo" in cause_text:
        mitigations.append(
            "Include more clinical photography datasets or apply cross-domain "
            "adaptation techniques to bridge the dermoscopic-clinical gap."
        )

    if not mitigations:
        mitigations.append(
            "Ensemble multiple model types (logistic + XGBoost + deep MLP) to "
            "reduce individual model failure modes."
        )

    return mitigations


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Error analysis of trained SkinTag models")
    parser.add_argument("--model", type=str, default="deep",
                        help="Model to analyze: baseline, logistic, xgboost, deep")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of worst mispredictions to detail")
    args = parser.parse_args()

    import yaml
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cache_dir = PROJECT_ROOT / "results" / "cache"
    seed = config["training"]["seed"]

    # Load data
    meta_path = cache_dir / "metadata.csv"
    emb_path = cache_dir / "embeddings.pt"
    if not meta_path.exists() or not emb_path.exists():
        print("Run the pipeline first: python run_pipeline.py --no-app")
        return

    all_meta = pd.read_csv(meta_path)
    embeddings = torch.load(emb_path, weights_only=True)
    labels_all = all_meta["label"].values

    # Reconstruct test split
    indices = np.arange(len(all_meta))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels_all)

    X_test = embeddings[test_idx].numpy()
    y_test = labels_all[test_idx]
    test_meta = all_meta.iloc[test_idx].reset_index(drop=True)

    # Load model
    model_path = cache_dir / f"classifier_{args.model}.pkl"
    if not model_path.exists():
        model_path = cache_dir / "classifier.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    print(f"Model: {args.model} ({model_path.name})")
    print(f"Test samples: {len(y_test)}")

    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    if y_proba.ndim == 2:
        mal_proba = y_proba[:, 1]
    else:
        mal_proba = y_proba

    # Find mispredictions
    wrong_mask = y_pred != y_test
    n_wrong = wrong_mask.sum()
    n_total = len(y_test)
    print(f"Mispredictions: {n_wrong}/{n_total} ({n_wrong/n_total*100:.1f}%)")

    if n_wrong == 0:
        print("No errors found!")
        return

    # Rank by confidence of wrong prediction (highest confidence wrong = worst)
    wrong_indices = np.where(wrong_mask)[0]
    wrong_confidence = np.abs(mal_proba[wrong_indices] - 0.5) + 0.5  # distance from boundary
    sorted_order = np.argsort(-wrong_confidence)  # highest confidence first
    top_k = min(args.top, len(sorted_order))

    # Error type breakdown
    fn_mask = (y_test == 1) & (y_pred == 0)
    fp_mask = (y_test == 0) & (y_pred == 1)
    print(f"\nError breakdown:")
    print(f"  False negatives (missed malignant): {fn_mask.sum()} ({fn_mask.sum()/n_total*100:.1f}%)")
    print(f"  False positives (false alarm):      {fp_mask.sum()} ({fp_mask.sum()/n_total*100:.1f}%)")

    # Detailed top-K analysis
    print(f"\n{'='*70}")
    print(f"TOP {top_k} WORST MISPREDICTIONS (highest confidence errors)")
    print(f"{'='*70}")

    error_details = []
    all_causes = []

    # Create output directory for error images
    error_img_dir = cache_dir / "error_analysis_images"
    error_img_dir.mkdir(parents=True, exist_ok=True)

    for rank, order_idx in enumerate(sorted_order[:top_k]):
        test_idx_local = wrong_indices[order_idx]
        row = test_meta.iloc[test_idx_local]

        true_label = int(y_test[test_idx_local])
        pred_label = int(y_pred[test_idx_local])
        confidence = float(mal_proba[test_idx_local])
        true_name = "MALIGNANT" if true_label == 1 else "BENIGN"
        pred_name = "MALIGNANT" if pred_label == 1 else "BENIGN"

        # Condition info
        condition_name = "Unknown"
        if "condition_label" in row.index:
            try:
                cond = Condition(int(float(row["condition_label"])))
                condition_name = CONDITION_NAMES.get(cond, "Unknown")
            except (ValueError, TypeError):
                pass

        # Root cause analysis
        causes = _classify_error_type(true_label, pred_label, confidence, row)
        all_causes.append(causes)

        # Image path
        image_path = str(row.get("image_path", "N/A"))

        # Copy image to error analysis dir
        if image_path != "N/A" and Path(image_path).exists():
            dest = error_img_dir / f"error_{rank+1}_{Path(image_path).name}"
            try:
                shutil.copy2(image_path, dest)
            except Exception:
                pass

        detail = {
            "rank": rank + 1,
            "true_label": true_name,
            "predicted_label": pred_name,
            "malignancy_probability": round(confidence, 4),
            "condition": condition_name,
            "dataset": str(row.get("dataset", "unknown")),
            "domain": str(row.get("domain", "unknown")),
            "fitzpatrick": str(row.get("fitzpatrick", "unknown")),
            "image_path": image_path,
            "root_causes": causes,
        }
        error_details.append(detail)

        print(f"\n  #{rank+1}")
        print(f"  True: {true_name}  |  Predicted: {pred_name}  |  P(malignant)={confidence:.3f}")
        print(f"  Condition: {condition_name}  |  Dataset: {detail['dataset']}  |  Domain: {detail['domain']}")
        if detail["fitzpatrick"] != "unknown":
            print(f"  Fitzpatrick: {detail['fitzpatrick']}")
        print(f"  Image: {image_path}")
        print(f"  Root causes:")
        for cause in causes:
            print(f"    - {cause}")

    # Mitigations
    mitigations = _suggest_mitigations(all_causes)
    print(f"\n{'='*70}")
    print("PROPOSED MITIGATIONS")
    print(f"{'='*70}")
    for i, m in enumerate(mitigations, 1):
        print(f"  {i}. {m}")

    # Domain breakdown of errors
    print(f"\n{'='*70}")
    print("ERROR RATE BY DOMAIN")
    print(f"{'='*70}")
    if "domain" in test_meta.columns:
        for domain in sorted(test_meta["domain"].unique()):
            d_mask = test_meta["domain"].values == domain
            d_wrong = wrong_mask[d_mask].sum()
            d_total = d_mask.sum()
            d_fn = fn_mask[d_mask].sum()
            rate = d_wrong / d_total * 100 if d_total > 0 else 0
            print(f"  {domain:<15} errors={d_wrong}/{d_total} ({rate:.1f}%)  "
                  f"false_neg={d_fn}")

    # Fitzpatrick breakdown of errors
    if "fitzpatrick" in test_meta.columns:
        print(f"\n{'='*70}")
        print("ERROR RATE BY FITZPATRICK SKIN TYPE")
        print(f"{'='*70}")
        fitz_col = test_meta["fitzpatrick"].fillna("unknown").astype(str)
        for ftype in sorted(fitz_col.unique()):
            f_mask = fitz_col.values == ftype
            f_wrong = wrong_mask[f_mask].sum()
            f_total = f_mask.sum()
            f_fn = fn_mask[f_mask].sum()
            rate = f_wrong / f_total * 100 if f_total > 0 else 0
            print(f"  FST {ftype:<10} errors={f_wrong}/{f_total} ({rate:.1f}%)  "
                  f"false_neg={f_fn}")

    # Save structured results
    output = {
        "model": args.model,
        "total_test": n_total,
        "total_errors": int(n_wrong),
        "error_rate": round(n_wrong / n_total, 4),
        "false_negatives": int(fn_mask.sum()),
        "false_positives": int(fp_mask.sum()),
        "top_errors": error_details,
        "mitigations": mitigations,
    }

    out_path = cache_dir / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Error images copied to {error_img_dir}/")


if __name__ == "__main__":
    main()
