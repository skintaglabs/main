#!/usr/bin/env python3
"""Complete the interrupted pipeline by training classifiers on saved fine-tuned embeddings.

The full_retraining_pipeline.py run from 2026-02-04 completed SigLIP fine-tuning
and embedding extraction but crashed before training classifiers on the fine-tuned
embeddings. This script picks up where it left off.

Usage:
    python scripts/complete_finetuned_classifiers.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths from the interrupted run
RUN_DIR = PROJECT_ROOT / "results" / "cache" / "clinical_v20260204_161741"
CACHE_DIR = PROJECT_ROOT / "results" / "cache"
FT_DIR = RUN_DIR / "siglip_finetuned"
OUTPUT_DIR = RUN_DIR / "classifiers_finetuned"


def train_xgboost(X_train, y_train, X_test, y_test, n_classes=2):
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    class_counts = np.bincount(y_train)
    class_weights = {i: len(y_train) / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    if n_classes == 2:
        class_weights[1] *= 2.0

    sample_weights = np.array([class_weights[y] for y in y_train])

    clf = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="auc" if n_classes == 2 else "mlogloss",
        tree_method="hist", device="cuda",
        random_state=42,
    )
    clf.fit(X_train_s, y_train, sample_weight=sample_weights)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    return clf, scaler, {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "auc": float(auc),
    }


def train_mlp(X_train, y_train, X_test, y_test, n_classes=2):
    import torch.nn as nn

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_train_s.shape[1]

    model = nn.Sequential(
        nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, n_classes),
    ).to(device)

    X_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0], device=device) if n_classes == 2 else None
    )

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    if n_classes == 2:
        auc = roc_auc_score(y_test, probs[:, 1])
    else:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")

    return model, scaler, {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro")),
        "auc": float(auc),
    }


def main():
    print("=" * 60)
    print("COMPLETING INTERRUPTED PIPELINE")
    print("=" * 60)
    print("Note: Train embeddings were shuffled by WeightedRandomSampler.")
    print("Using TEST embeddings only (correctly ordered) with train/eval split.")

    # Verify fine-tuned embeddings exist
    test_emb_path = FT_DIR / "embeddings_test.pt"
    if not test_emb_path.exists():
        print(f"Error: Fine-tuned test embeddings not found in {FT_DIR}")
        sys.exit(1)

    # Load fine-tuned TEST embeddings (correctly ordered, no sampler)
    print("\nLoading fine-tuned test embeddings...")
    X_test_ft = torch.load(test_emb_path, weights_only=True).numpy()
    print(f"  Test embeddings: {X_test_ft.shape}")

    # Load and align labels
    full_meta = pd.read_csv(CACHE_DIR / "metadata.csv")
    test_meta_csv = pd.read_csv(CACHE_DIR / "test_metadata.csv")

    test_ids = set(test_meta_csv["sample_id"])
    test_df = full_meta[full_meta["sample_id"].isin(test_ids)].reset_index(drop=True)

    # Filter to samples with valid image paths
    data_dir = PROJECT_ROOT / "data"
    path_patterns = {
        "ham10000": [data_dir / "Skin Cancer" / "Skin Cancer" / "{}.jpg"],
        "ddi": [data_dir / "ddi" / "images" / "{}.jpg", data_dir / "ddi" / "images" / "{}.png"],
        "fitzpatrick17k": [data_dir / "fitzpatrick17k" / "images" / "{}.jpg", data_dir / "fitzpatrick17k" / "images" / "{}.png"],
        "pad_ufes": [data_dir / "pad_ufes" / "images" / "{}.png", data_dir / "pad_ufes" / "images" / "{}.jpg"],
        "bcn20000": [data_dir / "bcn20000" / "images" / "{}.jpg", data_dir / "bcn20000" / "images" / "{}.JPG"],
    }

    valid_mask = []
    for _, row in test_df.iterrows():
        found = any(
            Path(str(p).format(row["sample_id"])).exists()
            for p in path_patterns.get(row["dataset"], [])
        )
        valid_mask.append(found)

    test_filtered = test_df[np.array(valid_mask)].reset_index(drop=True)
    n_ft = X_test_ft.shape[0]

    print(f"  Filtered test samples: {len(test_filtered)} (embeddings: {n_ft})")
    if len(test_filtered) != n_ft:
        print(f"  Warning: Count mismatch, truncating")
        test_filtered = test_filtered.iloc[:n_ft]

    y_binary = test_filtered["label"].values
    y_condition = test_filtered["condition_label"].values
    print(f"  Malignant: {y_binary.sum()}/{len(y_binary)}")

    # Stratified split: 70% train, 30% eval
    np.random.seed(42)
    indices = np.random.permutation(n_ft)
    split = int(n_ft * 0.7)
    train_idx, eval_idx = indices[:split], indices[split:]

    X_train_ft = X_test_ft[train_idx]
    X_eval_ft = X_test_ft[eval_idx]
    y_train_binary = y_binary[train_idx]
    y_eval_binary = y_binary[eval_idx]
    y_train_condition = y_condition[train_idx]
    y_eval_condition = y_condition[eval_idx]

    # Rename for downstream code
    y_test_binary = y_eval_binary
    y_test_condition = y_eval_condition
    X_test_eval = X_eval_ft

    print(f"  Train split: {len(train_idx)}, Eval split: {len(eval_idx)}")
    print(f"  Train malignant: {y_train_binary.sum()}, Eval malignant: {y_eval_binary.sum()}")

    # Also load frozen embeddings for comparison (same test subset)
    frozen_emb_path = CACHE_DIR / "embeddings.pt"
    frozen_results = {}
    if frozen_emb_path.exists():
        print("\nLoading frozen embeddings for comparison...")
        all_frozen = torch.load(frozen_emb_path, weights_only=True).numpy()
        sid_to_idx = {sid: idx for idx, sid in enumerate(full_meta["sample_id"])}
        # Get frozen embeddings for the SAME filtered test samples
        frozen_test_idx = [sid_to_idx[sid] for sid in test_filtered["sample_id"]]
        X_frozen_all = all_frozen[frozen_test_idx]
        # Use same train/eval split
        X_frozen_train = X_frozen_all[train_idx]
        X_frozen_eval = X_frozen_all[eval_idx]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ===== XGBoost on fine-tuned embeddings =====
    print("\n[1/4] XGBoost on fine-tuned embeddings (binary)...")
    xgb_clf, xgb_scaler, xgb_res = train_xgboost(
        X_train_ft, y_train_binary, X_eval_ft, y_eval_binary, n_classes=2
    )
    results["xgboost_finetuned_binary"] = xgb_res
    print(f"      AUC: {xgb_res['auc']:.4f}, Acc: {xgb_res['accuracy']:.1%}, F1: {xgb_res['f1']:.3f}")
    with open(OUTPUT_DIR / "xgboost_finetuned_binary.pkl", "wb") as f:
        pickle.dump({"classifier": xgb_clf, "scaler": xgb_scaler}, f)

    # ===== MLP on fine-tuned embeddings =====
    print("\n[2/4] MLP on fine-tuned embeddings (binary)...")
    mlp_clf, mlp_scaler, mlp_res = train_mlp(
        X_train_ft, y_train_binary, X_eval_ft, y_eval_binary, n_classes=2
    )
    results["mlp_finetuned_binary"] = mlp_res
    print(f"      AUC: {mlp_res['auc']:.4f}, Acc: {mlp_res['accuracy']:.1%}, F1: {mlp_res['f1']:.3f}")
    with open(OUTPUT_DIR / "mlp_finetuned_binary.pkl", "wb") as f:
        pickle.dump({"classifier": mlp_clf, "scaler": mlp_scaler}, f)

    # ===== XGBoost on fine-tuned embeddings (condition) =====
    print("\n[3/4] XGBoost on fine-tuned embeddings (10-class condition)...")
    xgb_cond, xgb_cond_scaler, xgb_cond_res = train_xgboost(
        X_train_ft, y_train_condition, X_eval_ft, y_eval_condition, n_classes=10
    )
    results["xgboost_finetuned_condition"] = xgb_cond_res
    print(f"      AUC: {xgb_cond_res['auc']:.4f}, Acc: {xgb_cond_res['accuracy']:.1%}, F1: {xgb_cond_res['f1']:.3f}")
    with open(OUTPUT_DIR / "xgboost_finetuned_condition.pkl", "wb") as f:
        pickle.dump({"classifier": xgb_cond, "scaler": xgb_cond_scaler}, f)

    # ===== Frozen comparison (same split) =====
    print("\n[4/4] XGBoost on frozen embeddings (for comparison)...")
    if frozen_emb_path.exists():
        _, _, frozen_res = train_xgboost(
            X_frozen_train, y_train_binary,
            X_frozen_eval, y_eval_binary, n_classes=2
        )
        frozen_results["xgboost_frozen_binary"] = frozen_res
        print(f"      AUC: {frozen_res['auc']:.4f}, Acc: {frozen_res['accuracy']:.1%}, F1: {frozen_res['f1']:.3f}")

    # ===== Clinical thresholds on best model =====
    print("\n" + "=" * 60)
    print("CLINICAL SENSITIVITY/SPECIFICITY (XGBoost fine-tuned)")
    print("=" * 60)

    X_eval_s = xgb_scaler.transform(X_eval_ft)
    y_prob = xgb_clf.predict_proba(X_eval_s)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_eval_binary, y_prob)

    clinical = {}
    for target in [0.99, 0.95, 0.90, 0.85]:
        idx = np.argmin(np.abs(tpr - target))
        thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        spec = float(1 - fpr[idx])
        clinical[f"sens_{int(target*100)}"] = {"threshold": thresh, "specificity": spec}
        print(f"  At {target:.0%} sensitivity: threshold={thresh:.3f}, specificity={spec:.1%}")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("FROZEN vs FINE-TUNED COMPARISON")
    print("=" * 60)
    print(f"{'Model':<35} {'AUC':>10} {'Accuracy':>12} {'F1':>10}")
    print("-" * 67)

    if frozen_results:
        fr = frozen_results["xgboost_frozen_binary"]
        print(f"{'XGBoost (frozen embeddings)':<35} {fr['auc']:>10.4f} {fr['accuracy']:>11.1%} {fr['f1']:>10.3f}")

    ft = results["xgboost_finetuned_binary"]
    print(f"{'XGBoost (fine-tuned embeddings)':<35} {ft['auc']:>10.4f} {ft['accuracy']:>11.1%} {ft['f1']:>10.3f}")

    ml = results["mlp_finetuned_binary"]
    print(f"{'MLP (fine-tuned embeddings)':<35} {ml['auc']:>10.4f} {ml['accuracy']:>11.1%} {ml['f1']:>10.3f}")

    cond = results["xgboost_finetuned_condition"]
    print(f"{'XGBoost condition (fine-tuned)':<35} {cond['auc']:>10.4f} {cond['accuracy']:>11.1%} {cond['f1']:>10.3f}")

    if frozen_results:
        improvement = (ft["auc"] - fr["auc"]) / fr["auc"] * 100
        print(f"\nFine-tuning improvement: {improvement:+.2f}% AUC")

    # Save results
    all_results = {
        "finetuned_classifiers": results,
        "frozen_comparison": frozen_results,
        "clinical_thresholds": clinical,
        "siglip_config": json.load(open(FT_DIR / "config.json")),
    }
    with open(RUN_DIR / "finetuned_classifier_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RUN_DIR / 'finetuned_classifier_results.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
