"""Cross-domain evaluation — the focused experiment.

Leave-one-domain-out cross-validation:
  Train on dermoscopic+clinical, test on smartphone, etc.
  Measures domain generalization gap with and without augmentations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import pickle
import numpy as np
import pandas as pd
import torch

from src.model.embeddings import EmbeddingExtractor
from src.model.classifier import SklearnClassifier
from src.model.baseline import MajorityClassBaseline
from src.model.deep_classifier import DeepClassifier
from src.data.loader import load_multi_dataset, get_demographic_groups
from src.data.schema import samples_to_arrays
from src.data.sampler import compute_domain_balanced_weights
from src.evaluation.metrics import robustness_report, cross_domain_report


def run_experiment(
    embeddings, labels, metadata,
    model_type="logistic",
    domain_balance=False,
    seed=42,
    device="cpu",
):
    """Run leave-one-domain-out evaluation for a single model config."""
    domains = metadata["domain"].values
    unique_domains = np.unique(domains)
    results = {}

    for held_out in unique_domains:
        test_mask = domains == held_out
        train_mask = ~test_mask

        if test_mask.sum() < 5 or train_mask.sum() < 5:
            print(f"  Skipping {held_out}: too few samples (train={train_mask.sum()}, test={test_mask.sum()})")
            continue

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_test = embeddings[test_mask]
        y_test = labels[test_mask]

        # Domain-balanced weights
        sample_weights = None
        if domain_balance:
            train_domains = domains[train_mask]
            sample_weights = compute_domain_balanced_weights(train_domains, y_train)

        # Train model
        if model_type == "baseline":
            clf = MajorityClassBaseline()
            clf.fit(X_train, y_train)
        elif model_type == "logistic":
            clf = SklearnClassifier(classifier_type="logistic")
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        elif model_type == "xgboost":
            clf = SklearnClassifier(classifier_type="xgboost")
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        elif model_type == "deep":
            clf = DeepClassifier(embedding_dim=X_train.shape[1], device=device)
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            continue

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

        # Per-group metrics for test domain
        test_meta = metadata[test_mask].reset_index(drop=True)
        groups = get_demographic_groups(test_meta)

        report = robustness_report(
            y_test, y_pred,
            groups=groups if groups else None,
            class_names=["benign", "malignant"],
            y_proba=y_proba,
        )

        results[held_out] = {
            "accuracy": report["overall_accuracy"],
            "balanced_accuracy": report["balanced_accuracy"],
            "f1_macro": report["f1_macro"],
            "f1_binary": report["f1_binary"],
            "auc": report.get("auc", float("nan")),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }

        # Add Fitzpatrick breakdown if available
        if "per_fitzpatrick" in report:
            results[held_out]["per_fitzpatrick"] = report["per_fitzpatrick"]

        print(f"  Held out {held_out}: acc={report['overall_accuracy']:.3f}, "
              f"f1_macro={report['f1_macro']:.3f}, bal_acc={report['balanced_accuracy']:.3f}")

    # Domain generalization gap
    if len(results) >= 2:
        accs = [r["accuracy"] for r in results.values()]
        f1s = [r["f1_macro"] for r in results.values()]
        results["_summary"] = {
            "mean_accuracy": float(np.mean(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "domain_gap": float(max(accs) - min(accs)),
            "domain_f1_gap": float(max(f1s) - min(f1s)),
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Sample N images per dataset (0=all)")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = PROJECT_ROOT / "data"
    cache_dir = PROJECT_ROOT / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config["extraction"]["batch_size_gpu"] if device == "cuda" else config["extraction"]["batch_size_cpu"]
    seed = config["training"]["seed"]

    # Load multi-dataset
    datasets = args.datasets or config.get("data", {}).get("datasets", None)
    samples = load_multi_dataset(data_dir, datasets=datasets)

    if not samples:
        print("No samples loaded. Ensure datasets are available in data/ directory.")
        return

    images, labels, metadata = samples_to_arrays(samples)

    # Sample if requested
    if args.sample > 0 and args.sample < len(images):
        np.random.seed(seed)
        indices = np.random.choice(len(images), args.sample, replace=False)
        images = [images[i] for i in indices]
        labels = labels[indices]
        metadata = metadata.iloc[indices].reset_index(drop=True)

    # Extract embeddings
    cache_path = cache_dir / "embeddings_multi.pt"
    extractor = EmbeddingExtractor(device=device)
    embeddings = extractor.extract_dataset(images, batch_size=batch_size, cache_path=cache_path)
    extractor.unload_model()
    embeddings_np = embeddings.numpy()

    print(f"\nTotal samples: {len(labels)}")
    print(f"Domains: {dict(zip(*np.unique(metadata['domain'], return_counts=True)))}")

    # Run experiments
    all_experiments = {}
    model_types = ["baseline", "logistic", "xgboost", "deep"]
    balance_modes = [False, True]

    for model_type in model_types:
        for balanced in balance_modes:
            name = f"{model_type}_{'balanced' if balanced else 'unbalanced'}"
            print(f"\n{'='*60}")
            print(f"Experiment: {name}")
            print(f"{'='*60}")

            results = run_experiment(
                embeddings_np, labels, metadata,
                model_type=model_type,
                domain_balance=balanced,
                seed=seed,
                device=device,
            )
            all_experiments[name] = results

    # Summary table
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<35} {'Mean Acc':>10} {'Mean F1':>10} {'Acc Gap':>10} {'F1 Gap':>10}")
    print("-" * 78)
    for name, results in all_experiments.items():
        summary = results.get("_summary", {})
        mean_acc = summary.get("mean_accuracy", float("nan"))
        mean_f1 = summary.get("mean_f1_macro", float("nan"))
        gap = summary.get("domain_gap", float("nan"))
        f1_gap = summary.get("domain_f1_gap", float("nan"))
        print(f"{name:<35} {mean_acc:>10.3f} {mean_f1:>10.3f} {gap:>10.3f} {f1_gap:>10.3f}")

    # Save
    output_path = cache_dir / "cross_domain_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_experiments, default=convert))
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
