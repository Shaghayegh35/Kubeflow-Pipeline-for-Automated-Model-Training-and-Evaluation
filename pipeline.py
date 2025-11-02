"""
Kubeflow Pipelines v2: minimal classification pipeline.

Steps:
  prep -> train -> evaluate

Usage:
  pip install -r requirements.txt
  python pipeline.py --compile   # writes pipeline.json
"""
from __future__ import annotations
import argparse, os, joblib
from kfp import dsl
from kfp.dsl import component, Dataset, Model, Metrics

@component(base_image="python:3.11")
def prep(data_out: Dataset):
    """Synthesize a small numeric dataset and write to CSV artifacts."""
    import pandas as pd, numpy as np, os
    os.makedirs(data_out.path, exist_ok=True)
    X = np.random.rand(1000, 8)
    y = (X.sum(axis=1) > 4).astype(int)
    pd.DataFrame(X).to_csv(os.path.join(data_out.path, 'X.csv'), index=False)
    pd.DataFrame(y).to_csv(os.path.join(data_out.path, 'y.csv'), index=False)

@component(base_image="python:3.11")
def train(data_in: Dataset, model_out: Model, n_estimators: int = 200, seed: int = 42):
    """Train a RandomForestClassifier and write model artifact."""
    import pandas as pd, os, joblib
    from sklearn.ensemble import RandomForestClassifier
    X = pd.read_csv(os.path.join(data_in.path,'X.csv')).values
    y = pd.read_csv(os.path.join(data_in.path,'y.csv')).values.ravel()
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed).fit(X,y)
    os.makedirs(model_out.path, exist_ok=True)
    joblib.dump(clf, os.path.join(model_out.path, 'rf.joblib'))

@component(base_image="python:3.11")
def evaluate(model_in: Model, data_in: Dataset, metrics_out: Metrics):
    """Evaluate F1 on the synthetic dataset and log as a pipeline metric."""
    import pandas as pd, os, joblib
    from sklearn.metrics import f1_score
    X = pd.read_csv(os.path.join(data_in.path,'X.csv')).values
    y = pd.read_csv(os.path.join(data_in.path,'y.csv')).values.ravel()
    clf = joblib.load(os.path.join(model_in.path, 'rf.joblib'))
    pred = clf.predict(X)
    f1 = f1_score(y, pred)
    metrics_out.log_metric("f1", float(f1))

@dsl.pipeline(name="minimal-kfp-pipeline")
def pipe(n_estimators: int = 200, seed: int = 42):
    d = prep()
    m = train(d.outputs["data_out"], n_estimators=n_estimators, seed=seed)
    e = evaluate(m.outputs["model_out"], d.outputs["data_out"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="Compile the pipeline to pipeline.json")
    args = ap.parse_args()
    if args.compile:
        from kfp import compiler
        compiler.Compiler().compile(pipe, package_path="pipeline.json")
        print("Wrote pipeline.json")
