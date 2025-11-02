import argparse, os, joblib, numpy as np
from kfp import dsl
from kfp.dsl import component, Dataset, Model, Metrics

@component(base_image="python:3.11")
def prep(data_out: Dataset):
    import pandas as pd, numpy as np, os
    os.makedirs(data_out.path, exist_ok=True)
    X = np.random.rand(1000, 8)
    y = (X.sum(axis=1) > 4).astype(int)
    pd.DataFrame(X).to_csv(os.path.join(data_out.path, 'X.csv'), index=False)
    pd.DataFrame(y).to_csv(os.path.join(data_out.path, 'y.csv'), index=False)

@component(base_image="python:3.11")
def train(data_in: Dataset, model_out: Model):
    import pandas as pd, os, joblib
    from sklearn.ensemble import RandomForestClassifier
    X = pd.read_csv(os.path.join(data_in.path,'X.csv')).values
    y = pd.read_csv(os.path.join(data_in.path,'y.csv')).values.ravel()
    clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X,y)
    os.makedirs(model_out.path, exist_ok=True)
    joblib.dump(clf, os.path.join(model_out.path, 'rf.joblib'))

@component(base_image="python:3.11")
def evaluate(model_in: Model, data_in: Dataset, metrics_out: Metrics):
    import pandas as pd, os, joblib
    from sklearn.metrics import f1_score
    X = pd.read_csv(os.path.join(data_in.path,'X.csv')).values
    y = pd.read_csv(os.path.join(data_in.path,'y.csv')).values.ravel()
    clf = joblib.load(os.path.join(model_in.path, 'rf.joblib'))
    pred = clf.predict(X)
    f1 = f1_score(y, pred)
    metrics_out.log_metric("f1", float(f1))

@dsl.pipeline(name="minimal-kfp-pipeline")
def pipe():
    d = prep()
    m = train(d.outputs["data_out"])
    e = evaluate(m.outputs["model_out"], d.outputs["data_out"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    if args.compile:
        from kfp import compiler
        compiler.Compiler().compile(pipe, package_path="pipeline.json")
        print("Wrote pipeline.json")
