"""Hugging Face model download utilities.

Handles downloading model artifacts from Hugging Face Hub with caching.
"""

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download


def _get_cache_dir(cache_subdir: str) -> Path:
    """Get cache directory from environment or default."""
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / cache_subdir


def _get_token(token: Optional[str]) -> Optional[str]:
    """Get token from parameter or environment."""
    return token or os.getenv("HF_TOKEN")


def _is_v2(revision: Optional[str] = None) -> bool:
    """Check if the target revision is a v2 model."""
    revision = revision or os.getenv("HF_REVISION")
    return bool(revision and "v2" in revision)


def download_model_from_hf(
    repo_id: str,
    filename: str,
    cache_subdir: str = "models",
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Download model file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., "org/model-name")
        filename: File to download from the repository
        cache_subdir: Subdirectory within HF_HOME for caching
        token: Hugging Face API token (optional, for private repos)
        revision: Git revision (tag, branch, or commit hash)

    Returns:
        Path to the downloaded/cached model file
    """
    revision = revision or os.getenv("HF_REVISION")
    rev_info = f" (revision: {revision})" if revision else ""
    print(f"Downloading {repo_id}/{filename}{rev_info} from Hugging Face...")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(_get_cache_dir(cache_subdir)),
        token=_get_token(token),
    )

    print(f"Model cached at: {model_path}")
    return Path(model_path)


def download_e2e_model_from_hf(
    repo_id: str,
    cache_subdir: str = "models",
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Download end-to-end fine-tuned model directory from Hugging Face Hub.

    Downloads all model files and returns the directory path.

    v1 (main/v1-original): config.json, model_state.pt, head_state.pt
    v2 (v2-field-augmented): v2/config.json, v2/siglip_finetuned.pt, v2/classifiers/*

    Args:
        repo_id: Hugging Face repository ID (e.g., "skintaglabs/siglip-skin-lesion-classifier")
        cache_subdir: Subdirectory within HF_HOME for caching
        token: Hugging Face API token (optional, for private repos)
        revision: Git revision (tag, branch, or commit hash)

    Returns:
        Path to the downloaded model directory
    """
    revision = revision or os.getenv("HF_REVISION")
    is_v2 = _is_v2(revision)
    rev_info = f" (revision: {revision})" if revision else ""
    print(f"Downloading fine-tuned model from {repo_id}{rev_info}...")

    if is_v2:
        # v2 stores everything under the v2/ prefix
        patterns = ["v2/*"]
    else:
        # v1 stores model files at the repo root
        patterns = ["config.json", "model_state.pt", "head_state.pt"]

    model_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(_get_cache_dir(cache_subdir)),
        token=_get_token(token),
        allow_patterns=patterns,
    )

    print(f"Model downloaded to: {model_dir}")
    return Path(model_dir)


def get_model_config():
    """Get model repository configuration.

    Returns dict with model repository settings from environment variables
    or defaults for SkinTag models. Automatically selects the correct file
    paths based on the HF_REVISION (v1 vs v2 layout).
    """
    revision = os.getenv("HF_REVISION")
    is_v2 = _is_v2(revision)

    if is_v2:
        default_classifier = "v2/classifiers/xgboost_finetuned_binary.pkl"
        default_condition = "v2/classifiers/xgboost_finetuned_condition.pkl"
    else:
        default_classifier = "Misc/classifier_deep_mlp.pkl"
        default_condition = "Misc/xgboost_finetuned_condition.pkl"

    return {
        "repo_id": os.getenv("HF_REPO_ID", "skintaglabs/siglip-skin-lesion-classifier"),
        "revision": revision,
        "classifier_filename": os.getenv("HF_CLASSIFIER_FILE", default_classifier),
        "condition_classifier_filename": os.getenv("HF_CONDITION_FILE", default_condition),
    }
