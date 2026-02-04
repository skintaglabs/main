"""Comprehensive evaluation script for NeurIPS paper.

Compares frozen vs fine-tuned SigLIP embeddings, measures robustness to distortions,
collects runtime metrics, and generates all comparison tables.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)
from tqdm import tqdm
import albumentations as A

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_multi_dataset
from src.model.embeddings import EmbeddingExtractor
from src.model.classifier import SklearnClassifier
import joblib
from src.model.deep_classifier import DeepClassifier, EndToEndSigLIP, DeepClassificationHead


def load_finetuned_model(model_dir: Path, device: str = "cuda"):
    """Load the fine-tuned SigLIP model."""
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model_state.pt"

    with open(config_path) as f:
        config = json.load(f)

    model = EndToEndSigLIP(
        model_name=config["model_name"],
        hidden_dim=config["hidden_dim"],
        n_classes=config["n_classes"],
        dropout=config["dropout"],
        unfreeze_layers=config["unfreeze_layers"],
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


class FinetunedEmbeddingExtractor:
    """Extract embeddings using the fine-tuned SigLIP backbone."""

    def __init__(self, model: EndToEndSigLIP, device: str = "cuda"):
        self.model = model
        self.device = device
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    @torch.no_grad()
    def extract_dataset(self, images, batch_size: int = 16, cache_path: Path = None):
        """Extract embeddings from fine-tuned backbone (before classification head)."""
        if cache_path and cache_path.exists():
            print(f"Loading cached fine-tuned embeddings from {cache_path}")
            return torch.load(cache_path)

        self.model.eval()
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting fine-tuned embeddings"):
            batch_paths = images[i:i + batch_size]
            batch_images = [Image.open(p).convert("RGB") if isinstance(p, (str, Path)) else p
                          for p in batch_paths]

            inputs = self.processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # Get features from backbone (before classification head)
            features = self.model.backbone.get_image_features(pixel_values=pixel_values)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output

            all_embeddings.append(features.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(all_embeddings, cache_path)
            print(f"Cached fine-tuned embeddings to {cache_path}")

        return all_embeddings


def get_distortion_transforms():
    """Get a dictionary of distortion transforms for robustness testing."""
    return {
        "none": A.Compose([]),
        "blur_light": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        "blur_heavy": A.GaussianBlur(blur_limit=(7, 11), p=1.0),
        "noise_light": A.GaussNoise(var_limit=(10, 30), p=1.0),
        "noise_heavy": A.GaussNoise(var_limit=(50, 100), p=1.0),
        "brightness_dark": A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.3), contrast_limit=0, p=1.0),
        "brightness_bright": A.RandomBrightnessContrast(brightness_limit=(0.3, 0.4), contrast_limit=0, p=1.0),
        "compression_light": A.ImageCompression(quality_lower=50, quality_upper=60, p=1.0),
        "compression_heavy": A.ImageCompression(quality_lower=10, quality_upper=20, p=1.0),
        "rotation_15": A.Rotate(limit=(15, 15), p=1.0),
        "rotation_45": A.Rotate(limit=(45, 45), p=1.0),
        "combined_realistic": A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.GaussNoise(var_limit=(10, 30), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=80, p=0.5),
        ]),
    }


def apply_distortion(image_path, transform):
    """Apply distortion transform to an image."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    if transform is not None:
        augmented = transform(image=img_array)
        img_array = augmented["image"]

    return Image.fromarray(img_array)


def evaluate_model(y_true, y_pred, y_prob=None):
    """Compute comprehensive metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_malignant": f1_score(y_true, y_pred, pos_label=1),
        "precision_malignant": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_malignant": recall_score(y_true, y_pred, pos_label=1, zero_division=0),  # sensitivity
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except:
            metrics["auc"] = None

    return metrics


def measure_inference_time(model, sample_input, n_runs: int = 100, warmup: int = 10):
    """Measure average inference time."""
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input.to(device))

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input.to(device))

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def get_model_size(model_or_path):
    """Get model size in MB."""
    if isinstance(model_or_path, (str, Path)):
        return Path(model_or_path).stat().st_size / (1024 * 1024)
    else:
        param_size = sum(p.numel() * p.element_size() for p in model_or_path.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model_or_path.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def main():
    """Run comprehensive evaluation."""
    print("=" * 70)
    print("  Comprehensive Evaluation for NeurIPS Paper")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cache_dir = Path("results/cache")
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "performance": {},
        "robustness": {},
        "runtime": {},
        "model_sizes": {},
    }

    # Load data
    print("\n[1] Loading data...")
    data_dir = Path("data")
    samples = load_multi_dataset(data_dir)

    # Split (same as training)
    from sklearn.model_selection import train_test_split
    train_samples, test_samples = train_test_split(
        samples, test_size=0.2, stratify=[s.label for s in samples], random_state=42
    )

    test_paths = [s.image_path for s in test_samples]
    y_test = np.array([s.label for s in test_samples])

    print(f"Test set: {len(test_samples)} samples")

    # Load frozen embeddings
    print("\n[2] Loading frozen SigLIP embeddings...")
    frozen_embeddings = torch.load(cache_dir / "embeddings.pt")

    # Get test indices (need to match the split)
    all_paths = [s.image_path for s in samples]
    test_indices = [all_paths.index(p) for p in test_paths]
    X_test_frozen = frozen_embeddings[test_indices].numpy()

    # Load fine-tuned model and extract embeddings
    print("\n[3] Loading fine-tuned SigLIP model...")
    finetuned_model, finetuned_config = load_finetuned_model(
        Path("models/finetuned_siglip"), device=device
    )

    print("\n[4] Extracting fine-tuned embeddings...")
    finetuned_extractor = FinetunedEmbeddingExtractor(finetuned_model, device=device)
    finetuned_embeddings = finetuned_extractor.extract_dataset(
        test_paths,
        batch_size=16,
        cache_path=cache_dir / "embeddings_finetuned_test.pt"
    )
    X_test_finetuned = finetuned_embeddings.numpy()

    # Load existing models
    print("\n[5] Loading trained models...")
    from src.model.baseline import MajorityClassBaseline

    baseline = MajorityClassBaseline()
    baseline.fit(None, y_test)  # Just needs labels to know majority class

    # Load sklearn classifiers (saved with joblib)
    logistic_frozen = SklearnClassifier(classifier_type="logistic")
    logistic_frozen.pipeline = joblib.load(cache_dir / "classifier_logistic.pkl")

    xgboost_frozen = SklearnClassifier(classifier_type="xgboost")
    xgboost_frozen.pipeline = joblib.load(cache_dir / "classifier_xgboost.pkl")

    # Skip deep classifier for now - focus on XGBoost comparison
    deep_frozen = None

    # Train XGBoost on fine-tuned embeddings
    print("\n[6] Training XGBoost on fine-tuned embeddings...")

    # Need training embeddings too
    train_paths = [s.image_path for s in train_samples]
    train_indices = [all_paths.index(p) for p in train_paths]
    X_train_frozen = frozen_embeddings[train_indices].numpy()
    y_train = np.array([s.label for s in train_samples])

    # Extract fine-tuned embeddings for training set
    finetuned_train_embeddings = finetuned_extractor.extract_dataset(
        train_paths,
        batch_size=16,
        cache_path=cache_dir / "embeddings_finetuned_train.pt"
    )
    X_train_finetuned = finetuned_train_embeddings.numpy()

    xgboost_finetuned = SklearnClassifier(classifier_type="xgboost")

    start_time = time.perf_counter()
    xgboost_finetuned.fit(X_train_finetuned, y_train)
    xgboost_finetuned_train_time = time.perf_counter() - start_time

    joblib.dump(xgboost_finetuned.pipeline, cache_dir / "classifier_xgboost_finetuned.pkl")
    print(f"  Training time: {xgboost_finetuned_train_time:.2f}s")

    # Evaluate all models
    print("\n[7] Evaluating all models...")

    models_to_eval = {
        "baseline": (baseline, None, "N/A"),
        "logistic_frozen": (logistic_frozen, X_test_frozen, "frozen"),
        "xgboost_frozen": (xgboost_frozen, X_test_frozen, "frozen"),
        # "deep_frozen": (deep_frozen, X_test_frozen, "frozen"),  # Skipped
        "xgboost_finetuned": (xgboost_finetuned, X_test_finetuned, "finetuned"),
        "siglip_finetuned": (finetuned_model, None, "end-to-end"),  # Special handling
    }

    for name, (model, X_test, emb_type) in models_to_eval.items():
        print(f"\n  Evaluating {name}...")

        if name == "baseline":
            y_pred = baseline.predict(np.zeros((len(y_test), 1)))
            y_prob = None
        elif name == "siglip_finetuned":
            # Direct inference with fine-tuned model
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

            y_pred = []
            y_prob = []

            finetuned_model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, len(test_paths), 16), desc="Fine-tuned inference"):
                    batch_paths = test_paths[i:i+16]
                    batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
                    inputs = processor(images=batch_images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)

                    logits = finetuned_model(pixel_values)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                    y_pred.extend(preds.cpu().numpy())
                    y_prob.extend(probs[:, 1].cpu().numpy())

            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)
        else:
            y_pred = model.predict(X_test)
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except:
                y_prob = None

        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics["embedding_type"] = emb_type
        results["performance"][name] = metrics

        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"    F1 Malignant: {metrics['f1_malignant']:.4f}")
        if metrics.get('auc'):
            print(f"    AUC: {metrics['auc']:.4f}")

    # Robustness evaluation
    print("\n[8] Robustness evaluation (distortions)...")

    distortions = get_distortion_transforms()

    # Test robustness for XGBoost frozen, XGBoost finetuned, and end-to-end finetuned
    robustness_models = {
        "xgboost_frozen": (xgboost_frozen, "frozen"),
        "xgboost_finetuned": (xgboost_finetuned, "finetuned"),
    }

    # Use a subset for distortion tests (faster)
    subset_size = min(1000, len(test_paths))
    subset_indices = np.random.choice(len(test_paths), subset_size, replace=False)
    subset_paths = [test_paths[i] for i in subset_indices]
    y_subset = y_test[subset_indices]

    frozen_extractor = EmbeddingExtractor(device=device)

    for distortion_name, transform in distortions.items():
        print(f"\n  Distortion: {distortion_name}")
        results["robustness"][distortion_name] = {}

        # Apply distortion and extract embeddings
        distorted_images = [apply_distortion(p, transform) for p in tqdm(subset_paths, desc="Applying distortion")]

        # Frozen embeddings
        frozen_extractor.load_model()
        distorted_frozen_emb = frozen_extractor.extract_dataset(distorted_images, batch_size=16)
        X_distorted_frozen = distorted_frozen_emb.numpy()

        # Fine-tuned embeddings
        distorted_finetuned_emb = []
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        with torch.no_grad():
            for i in range(0, len(distorted_images), 16):
                batch = distorted_images[i:i+16]
                inputs = processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                features = finetuned_model.backbone.get_image_features(pixel_values=pixel_values)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                distorted_finetuned_emb.append(features.cpu())

        X_distorted_finetuned = torch.cat(distorted_finetuned_emb, dim=0).numpy()

        # Evaluate models
        for model_name, (model, emb_type) in robustness_models.items():
            X_distorted = X_distorted_frozen if emb_type == "frozen" else X_distorted_finetuned
            y_pred = model.predict(X_distorted)
            try:
                y_prob = model.predict_proba(X_distorted)[:, 1]
            except:
                y_prob = None

            metrics = evaluate_model(y_subset, y_pred, y_prob)
            results["robustness"][distortion_name][model_name] = metrics

            print(f"    {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    frozen_extractor.unload_model()

    # Runtime measurements
    print("\n[9] Measuring inference times...")

    # Prepare sample inputs
    sample_image = Image.open(test_paths[0]).convert("RGB")
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    sample_input = processor(images=[sample_image], return_tensors="pt")["pixel_values"]

    # Fine-tuned model inference time
    finetuned_model.eval()
    finetuned_time = measure_inference_time(finetuned_model, sample_input, n_runs=50)
    results["runtime"]["siglip_finetuned_inference_ms"] = finetuned_time * 1000
    print(f"  Fine-tuned SigLIP inference: {finetuned_time*1000:.2f} ms")

    # Frozen embedding extraction time
    frozen_extractor.load_model()
    frozen_model = frozen_extractor.model

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            inputs = processor(images=[sample_image], return_tensors="pt").to(device)
            _ = frozen_model.vision_model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    frozen_emb_time = (time.perf_counter() - start) / 50
    results["runtime"]["frozen_embedding_extraction_ms"] = frozen_emb_time * 1000
    print(f"  Frozen embedding extraction: {frozen_emb_time*1000:.2f} ms")

    # XGBoost inference time (on CPU, very fast)
    sample_emb = np.random.randn(1, 1152).astype(np.float32)
    start = time.perf_counter()
    for _ in range(1000):
        _ = xgboost_frozen.predict(sample_emb)
    xgb_time = (time.perf_counter() - start) / 1000
    results["runtime"]["xgboost_inference_ms"] = xgb_time * 1000
    print(f"  XGBoost inference: {xgb_time*1000:.4f} ms")

    frozen_extractor.unload_model()

    # Model sizes
    print("\n[10] Collecting model sizes...")

    results["model_sizes"]["siglip_finetuned_full_mb"] = get_model_size(Path("models/finetuned_siglip/model_state.pt"))
    results["model_sizes"]["siglip_finetuned_head_mb"] = get_model_size(Path("models/finetuned_siglip/head_state.pt"))
    results["model_sizes"]["xgboost_frozen_mb"] = get_model_size(cache_dir / "classifier_xgboost.pkl")
    results["model_sizes"]["xgboost_finetuned_mb"] = get_model_size(cache_dir / "classifier_xgboost_finetuned.pkl")
    results["model_sizes"]["logistic_mb"] = get_model_size(cache_dir / "classifier_logistic.pkl")
    results["model_sizes"]["deep_mlp_mb"] = get_model_size(cache_dir / "classifier_deep.pkl")

    for name, size in results["model_sizes"].items():
        print(f"  {name}: {size:.2f} MB")

    # Training times
    results["runtime"]["xgboost_finetuned_train_s"] = xgboost_finetuned_train_time

    # Save results
    print("\n[11] Saving results...")

    output_path = output_dir / "comprehensive_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Print summary tables
    print("\n" + "=" * 70)
    print("  SUMMARY TABLES")
    print("=" * 70)

    print("\n[Table 1] Model Performance Comparison")
    print("-" * 70)
    print(f"{'Model':<25} {'Emb Type':<12} {'Acc':<8} {'F1 Mac':<8} {'F1 Mal':<8} {'AUC':<8}")
    print("-" * 70)
    for name, m in results["performance"].items():
        auc_str = f"{m['auc']:.4f}" if m.get('auc') else "N/A"
        print(f"{name:<25} {m['embedding_type']:<12} {m['accuracy']:.4f}   {m['f1_macro']:.4f}   {m['f1_malignant']:.4f}   {auc_str}")

    print("\n[Table 2] Robustness to Distortions (Accuracy)")
    print("-" * 70)
    print(f"{'Distortion':<25} {'XGB Frozen':<15} {'XGB Finetuned':<15}")
    print("-" * 70)
    for dist_name, dist_results in results["robustness"].items():
        frozen_acc = dist_results.get("xgboost_frozen", {}).get("accuracy", 0)
        finetuned_acc = dist_results.get("xgboost_finetuned", {}).get("accuracy", 0)
        print(f"{dist_name:<25} {frozen_acc:.4f}          {finetuned_acc:.4f}")

    print("\n[Table 3] Model Sizes and Inference Times")
    print("-" * 70)
    for name, size in results["model_sizes"].items():
        print(f"  {name}: {size:.2f} MB")
    print()
    for name, time_val in results["runtime"].items():
        print(f"  {name}: {time_val:.4f}")

    print("\n" + "=" * 70)
    print("  Evaluation complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
