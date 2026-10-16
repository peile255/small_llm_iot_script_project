from pathlib import Path
import argparse, yaml, json, sys, time
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import load_text
from src.metrics import compute_metrics, bootstrap_metric

def texts(df):
    return [load_text(p) for p in df.path]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    d = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(d / cfg["data"]["train_file"])
    test = pd.read_csv(d / cfg["data"]["test_file"])

    out = Path("outputs/baseline")
    out.mkdir(parents=True, exist_ok=True)

    char_range = tuple(cfg["baseline"]["char_ngram_range"])
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=char_range,
                                 min_df=2, max_features=cfg["baseline"]["max_features"])),
        ("word", TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[\w\-$.:\\/]+\b",
                                 ngram_range=(1,2), min_df=2, max_features=100000)),
    ])
    clf = LogisticRegression(
        C=cfg["baseline"]["C"], class_weight=cfg["baseline"]["class_weight"],
        max_iter=2000, random_state=cfg["seed"], n_jobs=-1
    )
    pipe = Pipeline([("features", feats), ("clf", clf)])
    t0 = time.perf_counter()
    pipe.fit(texts(train), train.label)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(texts(test))
    infer_s = time.perf_counter() - t0

    m = compute_metrics(test.label, pred)
    for key in ["accuracy","f1","fpr","mcc"]:
        m[f"{key}_ci95"] = bootstrap_metric(test.label, pred, key, cfg["bootstrap"]["n_resamples"], cfg["seed"])
    m["train_seconds"] = train_s
    m["median_latency_s"] = infer_s / max(1, len(test))
    m["throughput_files_s"] = len(test) / infer_s if infer_s else None

    pd.DataFrame({
        "sha256": test.sha256, "y_true": test.label, "y_pred": pred
    }).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
