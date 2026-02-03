# Model Hosting

Models hosted on Hugging Face Hub.

## Download Models

```bash
export USE_HF_MODELS=true
export HF_TOKEN=your_token  # For private repos
make app
```

Models auto-download and cache in `~/.cache/huggingface/skintag/`.

## Upload Models

```bash
pip install huggingface_hub[cli]
huggingface-cli login
huggingface-cli upload MedGemma540/skintag-models results/cache/classifier_deep_mlp.pkl
```

## Config

- `USE_HF_MODELS=true` - Enable HF downloads
- `HF_REPO_ID` - Override default repo (optional)
- `HF_TOKEN` - For private repos (optional)
