#!/usr/bin/env python3

from pathlib import Path
import json
import tarfile
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data import load_text
from src.metrics import compute_metrics


TRAIN_FILE = Path(
    "data/processed/final_frozen/final_train.csv"
)

EXTERNAL_FILE = Path(
    "outputs/paper_v3/frozen_external_v1/"
    "external_final_audited.csv"
)

ARCHIVE = Path(
    "data/external/wwe123/"
    "ast_ps1_by_label.tar.gz"
)

OUT_ROOT = Path(
    "outputs/paper_v3/external_baselines"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = {
    "lr_char_word": {
        "kind": "lr_char_word",
        "C": 2.0,
    },
    "lr_char": {
        "kind": "lr_char",
        "C": 2.0,
    },
    "svm_char": {
        "kind": "svm_char",
        "C": 1.0,
    },
    "svm_word": {
        "kind": "svm_word",
        "C": 2.0,
    },
}


def load_external_texts(df):
    wanted = set(
        df["archive_path"]
        .astype(str)
        .tolist()
    )

    text_map = {}

    with tarfile.open(
        ARCHIVE,
        mode="r|gz",
    ) as tf:

        for member in tf:

            if not member.isfile():
                continue

            if member.name not in wanted:
                continue

            f = tf.extractfile(member)

            if f is None:
                continue

            raw = f.read()

            # Match project text loading semantics.
            from src.data import decode_script

            text = decode_script(raw)

            text = (
                text.replace("\x00", "")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )

            text_map[
                member.name
            ] = text

            if (
                len(text_map) % 2000
                == 0
            ):
                print(
                    f"External loaded: "
                    f"{len(text_map)}/{len(wanted)}",
                    flush=True,
                )

    missing = wanted - set(text_map)

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} external files"
        )

    return [
        text_map[p]
        for p in df["archive_path"]
    ]


def make_model(name, C):

    if name == "lr_char_word":

        features = FeatureUnion([
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=200000,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=(
                        r"(?u)\b[\w\-$.:\\/]+\b"
                    ),
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100000,
                ),
            ),
        ])

        clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )

    elif name == "lr_char":

        features = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=200000,
        )

        clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )

    elif name == "svm_char":

        features = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=200000,
        )

        clf = LinearSVC(
            C=C,
            class_weight="balanced",
        )

    elif name == "svm_word":

        features = TfidfVectorizer(
            analyzer="word",
            token_pattern=(
                r"(?u)\b[\w\-$.:\\/]+\b"
            ),
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
        )

        clf = LinearSVC(
            C=C,
            class_weight="balanced",
        )

    else:
        raise ValueError(name)

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


def main():

    train = pd.read_csv(TRAIN_FILE)
    ext = pd.read_csv(EXTERNAL_FILE)

    assert len(train) == 4322
    assert len(ext) == 19632

    print(
        "Loading training texts...",
        flush=True,
    )

    train_texts = [
        load_text(p)
        for p in train["path"]
    ]

    print(
        "Loading external texts...",
        flush=True,
    )

    external_texts = (
        load_external_texts(ext)
    )

    y_train = train["label"].to_numpy()
    y_ext = ext["label"].to_numpy()

    summary_rows = []

    for model_name, spec in MODELS.items():

        print(
            f"\n=== {model_name} ===",
            flush=True,
        )

        out = OUT_ROOT / model_name
        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = make_model(
            spec["kind"],
            spec["C"],
        )

        t0 = time.perf_counter()

        model.fit(
            train_texts,
            y_train,
        )

        train_s = (
            time.perf_counter()
            - t0
        )

        t0 = time.perf_counter()

        pred = model.predict(
            external_texts
        )

        infer_s = (
            time.perf_counter()
            - t0
        )

        metrics = compute_metrics(
            y_ext,
            pred,
        )

        metrics.update({
            "model":
                model_name,
            "C":
                spec["C"],
            "n":
                int(len(ext)),
            "train_seconds":
                float(train_s),
            "inference_seconds":
                float(infer_s),
            "mean_latency_s":
                float(
                    infer_s / len(ext)
                ),
            "throughput_files_s":
                float(
                    len(ext) / infer_s
                ),
            "evaluation":
                "audited_external_full",
        })

        pred_df = pd.DataFrame({
            "external_final_index":
                ext["external_final_index"],
            "archive_path":
                ext["archive_path"],
            "raw_sha256":
                ext["raw_sha256"],
            "normalized_sha256":
                ext["normalized_sha256"],
            "y_true":
                y_ext,
            "y_pred":
                pred,
        })

        pred_df.to_csv(
            out /
            "external_predictions.csv",
            index=False,
        )

        (
            out /
            "external_metrics.json"
        ).write_text(
            json.dumps(
                metrics,
                indent=2,
            ) + "\n"
        )

        summary_rows.append(
            metrics
        )

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUT_ROOT /
        "external_baseline_summary.csv",
        index=False,
    )

    print(
        "\nSaved:",
        OUT_ROOT /
        "external_baseline_summary.csv",
    )


if __name__ == "__main__":
    main()
