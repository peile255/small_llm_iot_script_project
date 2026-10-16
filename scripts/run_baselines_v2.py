#!/usr/bin/env python3

from pathlib import Path
import json
import time
import yaml

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_text


CONFIG_PATH = Path("configs/v2/baselines_v2.yaml")
OUT_ROOT = Path("outputs/paper_v2/baselines")


def load_texts(df):
    return [load_text(p) for p in df["path"]]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "mcc": float(
            matthews_corrcoef(y_true, y_pred)
        ),
    }


def make_features(model_cfg):
    mode = model_cfg["feature_mode"]
    min_df = int(model_cfg.get("min_df", 2))

    char_vec = None
    word_vec = None

    if mode in {"char", "char_word"}:
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(
                model_cfg["char_ngram_range"]
            ),
            min_df=min_df,
            max_features=int(
                model_cfg["char_max_features"]
            ),
        )

    if mode in {"word", "char_word"}:
        word_vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[\w\-$.:\\/]+\b",
            ngram_range=tuple(
                model_cfg["word_ngram_range"]
            ),
            min_df=min_df,
            max_features=int(
                model_cfg["word_max_features"]
            ),
        )

    if mode == "char":
        return char_vec

    if mode == "word":
        return word_vec

    if mode == "char_word":
        return FeatureUnion([
            ("char", char_vec),
            ("word", word_vec),
        ])

    raise ValueError(f"Unknown feature_mode: {mode}")


def make_classifier(model_cfg, C, seed):
    classifier = model_cfg["classifier"]
    class_weight = model_cfg.get(
        "class_weight",
        "balanced",
    )

    if classifier == "logistic_regression":
        return LogisticRegression(
            C=float(C),
            class_weight=class_weight,
            max_iter=2000,
            random_state=seed,
            n_jobs=-1,
        )

    if classifier == "linear_svm":
        return LinearSVC(
            C=float(C),
            class_weight=class_weight,
            random_state=seed,
        )

    raise ValueError(
        f"Unknown classifier: {classifier}"
    )


def make_pipeline(model_cfg, C, seed):
    return Pipeline([
        ("features", make_features(model_cfg)),
        ("clf", make_classifier(model_cfg, C, seed)),
    ])


def save_predictions(path, df, pred):
    pd.DataFrame({
        "sha256": df["sha256"],
        "group_id": df["group_id"],
        "y_true": df["label"].astype(int),
        "y_pred": np.asarray(pred, dtype=int),
    }).to_csv(
        path,
        index=False,
    )


def main():
    cfg = yaml.safe_load(
        CONFIG_PATH.read_text()
    )

    seed = int(cfg["seed"])
    data_cfg = cfg["data"]
    droot = Path(data_cfg["processed_dir"])

    train = pd.read_csv(
        droot / data_cfg["train_file"]
    )
    val = pd.read_csv(
        droot / data_cfg["val_file"]
    )
    test = pd.read_csv(
        droot / data_cfg["test_file"]
    )

    print("Loading script text...")
    X_train = load_texts(train)
    X_val = load_texts(val)
    X_test = load_texts(test)

    y_train = train["label"].astype(int)
    y_val = val["label"].astype(int)
    y_test = test["label"].astype(int)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for model_name, model_cfg in cfg["models"].items():

        print()
        print("=" * 70)
        print("MODEL:", model_name)
        print("=" * 70)

        model_out = OUT_ROOT / model_name
        model_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Frozen primary baseline is not retuned.
        if model_name == "lr_char_word":
            candidate_C = [2.0]
            is_frozen_primary = True
        else:
            candidate_C = [
                float(x)
                for x in cfg["selection"]["C_values"]
            ]
            is_frozen_primary = False

        validation_rows = []

        for C in candidate_C:
            print(
                f"[validation] {model_name}, C={C}"
            )

            pipe = make_pipeline(
                model_cfg,
                C,
                seed,
            )

            t0 = time.perf_counter()
            pipe.fit(
                X_train,
                y_train,
            )
            train_seconds = (
                time.perf_counter() - t0
            )

            pred_val = pipe.predict(X_val)

            vm = compute_metrics(
                y_val,
                pred_val,
            )

            validation_rows.append({
                "model": model_name,
                "C": C,
                **vm,
                "train_seconds": train_seconds,
            })

            print(
                f"  val MCC={vm['mcc']:.6f} "
                f"F1={vm['f1']:.6f}"
            )

        validation_df = pd.DataFrame(
            validation_rows
        )

        validation_df.to_csv(
            model_out / "validation_search.csv",
            index=False,
        )

        if is_frozen_primary:
            selected_C = 2.0
            selection_note = (
                "Frozen primary reference; "
                "C=2.0 retained unchanged."
            )
        else:
            ranked = validation_df.sort_values(
                by=["mcc", "f1", "C"],
                ascending=[False, False, True],
            )

            selected_C = float(
                ranked.iloc[0]["C"]
            )

            selection_note = (
                "Selected using validation MCC; "
                "ties resolved by validation F1 "
                "then smaller C."
            )

        selected_config = {
            "model": model_name,
            "classifier":
                model_cfg["classifier"],
            "feature_mode":
                model_cfg["feature_mode"],
            "selected_C":
                selected_C,
            "selection_metric":
                cfg["selection"]["metric"],
            "selection_note":
                selection_note,
            "seed":
                seed,
            "final_training_partition":
                "train only",
            "validation_used_for":
                "hyperparameter selection only",
            "test_used_for":
                "final evaluation only",
        }

        (
            model_out /
            "selected_config.json"
        ).write_text(
            json.dumps(
                selected_config,
                indent=2,
            )
        )

        print(
            f"[test] {model_name}, "
            f"selected C={selected_C}"
        )

        final_pipe = make_pipeline(
            model_cfg,
            selected_C,
            seed,
        )

        t0 = time.perf_counter()

        final_pipe.fit(
            X_train,
            y_train,
        )

        final_train_seconds = (
            time.perf_counter() - t0
        )

        t0 = time.perf_counter()

        pred_test = final_pipe.predict(
            X_test
        )

        inference_seconds = (
            time.perf_counter() - t0
        )

        tm = compute_metrics(
            y_test,
            pred_test,
        )

        latency = (
            inference_seconds /
            max(1, len(test))
        )

        throughput = (
            len(test) /
            inference_seconds
            if inference_seconds > 0
            else None
        )

        test_metrics = {
            "model": model_name,
            "selected_C": selected_C,
            **tm,
            "train_seconds":
                final_train_seconds,
            "total_inference_seconds":
                inference_seconds,
            "mean_latency_s":
                latency,
            "throughput_files_s":
                throughput,
        }

        (
            model_out /
            "metrics.json"
        ).write_text(
            json.dumps(
                test_metrics,
                indent=2,
            )
        )

        save_predictions(
            model_out /
            "test_predictions.csv",
            test,
            pred_test,
        )

        summary_rows.append(
            test_metrics
        )

        print(
            f"  TEST MCC={tm['mcc']:.6f} "
            f"F1={tm['f1']:.6f} "
            f"Recall={tm['recall']:.6f}"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_ROOT /
        "baseline_v2_summary.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(
        OUT_ROOT /
        "baseline_v2_summary.csv"
    )


if __name__ == "__main__":
    main()
