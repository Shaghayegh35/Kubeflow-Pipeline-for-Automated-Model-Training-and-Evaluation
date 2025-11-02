# Kubeflow Pipeline — Automated Model Training & Evaluation (KFP v2)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![KFP](https://img.shields.io/badge/Kubeflow%20Pipelines-v2.7%2B-blue)
![CI](https://img.shields.io/badge/GitHub%20Actions-CI-brightgreen)
![License](https://img.shields.io/badge/License-MIT-success)

A production‑grade **Kubeflow Pipelines v2** example that automates the ML lifecycle:
**prep → train → evaluate**. It compiles to `pipeline.json` for upload to the KFP UI and
includes **tests** and **CI** for reliability.

> Updated 2025-11-02

---

## ✨ Features
- Modular KFP **components** with typed inputs/outputs
- **RandomForest** training and **F1** evaluation
- One‑command **compile** via CLI (`python pipeline.py --compile`)
- **Pytest** for sanity check (compiles pipeline)
- **GitHub Actions CI** (flake8 + tests + compile)
- Visual **diagram** of the pipeline

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python pipeline.py --compile
# Upload pipeline.json in the Kubeflow Pipelines UI
```

**Run tests:**
```bash
pytest -q
```

---

## 🧩 Components

| Step      | Purpose                                         | Outputs                            |
|-----------|--------------------------------------------------|------------------------------------|
| `prep`    | Synthesize numeric dataset, write CSV artifacts  | `X.csv`, `y.csv`                    |
| `train`   | Train RandomForest classifier                    | `rf.joblib`                         |
| `evaluate`| Compute F1 score                                 | Pipeline metric `f1`                |

---

## 📈 Example Output

- `pipeline.json` compiled successfully (upload to KFP UI)
- Example metric: **F1 ~ 0.80–0.90** (varies on synthetic data)

---

## 🛠️ CI (GitHub Actions)

- Lints with **flake8**
- Runs **pytest**
- Compiles `pipeline.json` on every push/PR

---

## 🗂 Project Layout
```
.
├─ pipeline.py
├─ requirements.txt
├─ tests/
│  └─ test_pipeline.py
├─ .github/workflows/ci.yml
├─ assets/
│  ├─ banner.png
│  └─ pipeline_diagram.png
└─ LICENSE
```

---

## 🖼 Visuals

<p align="center">
  <img src="assets/banner.png" width="80%"><br/>
  <img src="assets/pipeline_diagram.png" width="70%">
</p>

---

## 📄 License
MIT
