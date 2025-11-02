# Kubeflow Pipeline (KFP v2) — Minimal Classification Flow

![kfp](https://img.shields.io/badge/Kubeflow%20Pipelines-v2+-blue)

Components:
- `prep`: synthesize numeric dataset
- `train`: train RandomForest
- `evaluate`: log F1 metric
- Compile to `pipeline.json` for upload to KFP

> Created 2025-11-02

## Compile
```bash
pip install -r requirements.txt
python pipeline.py --compile
```
