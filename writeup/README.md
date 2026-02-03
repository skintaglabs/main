# SkinTag Paper Writeup

NeurIPS-style LaTeX paper for the SkinTag project.

## Files

- `main.tex` - Main paper document
- `references.bib` - BibTeX bibliography with all citations
- `neurips_2024.sty` - NeurIPS 2024 style file (simplified version)

## Building

```bash
# Build PDF (requires pdflatex and bibtex)
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or use Overleaf:
1. Upload all files to a new Overleaf project
2. For full NeurIPS formatting, replace `neurips_2024.sty` with the official version from https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles

## Paper Structure

1. **Introduction** - Problem statement, contributions
2. **Related Work** - Deep learning for dermatology, vision-language models, fairness in medical AI
3. **Datasets** - Five datasets (HAM10000, DDI, Fitzpatrick17k, PAD-UFES-20, BCN20000)
4. **Methods** - SigLIP fine-tuning, fairness-aware sampling
5. **Results** - Performance metrics, fairness analysis, cross-domain evaluation
6. **Discussion** - Insights, limitations, ethical considerations
7. **Conclusion** - Summary and future work

## Key Results

| Model | Accuracy | F1 Macro | F1 Malignant | AUC |
|-------|----------|----------|--------------|-----|
| Baseline | 79.1% | 0.442 | 0.000 | 0.500 |
| Logistic | 84.0% | 0.792 | 0.692 | 0.922 |
| XGBoost | 95.7% | 0.938 | 0.903 | 0.990 |
| **Fine-tuned SigLIP** | **92.3%** | **0.887** | **0.824** | — |

Fitzpatrick equalized odds sensitivity gap: **0.044** (< 5%)
