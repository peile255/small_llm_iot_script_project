#!/usr/bin/env python3

from pathlib import Path
import json
import tarfile
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data import load_text, decode_script
from src.metrics import compute_metrics


TRAIN_FILE = Path(
    "data/processed/final_frozen/final_train.csv"
)

EXTERNAL_FILE = Path(
    "outputs/paper_v4/threshold_sensitivity_v1/"
    "threshold_095_manifest.csv"
)

ARCHIVE = Path(
    "data/external/wwe123/"
    "ast_ps1_by_label.tar.gz"
)

OUT_ROOT = Path(
    "outputs/paper_v4/threshold_sensitivity_v1/"
    "threshold_095_predictions"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_external_texts(df):

    wanted = set(
        df["archive_path"].astype(str)
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

            text = decode_script(raw)

            text = (
                text
                .replace("\x00", "")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )

            text_map[member.name] = text

            if len(text_map) % 2000 == 0:
                print(
                    f"External loaded: "
                    f"{len(text_map)}/{len(wanted)}",
                    flush=True,
                )

    missing = wanted - set(text_map)

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} external scripts"
        )

    return [
        text_map[p]
        for p in df["archive_path"]
    ]


def save_result(
    model_name,
    C,
    ext,
    y_true,
    pred,
    train_seconds,
    inference_seconds,
):

    out = OUT_ROOT / model_name
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = compute_metrics(
        y_true,
        pred,
    )

    metrics.update({
        "model": model_name,
        "C": C,
        "n": int(len(ext)),
        "train_seconds":
            float(train_seconds),
        "classifier_predict_seconds":
            float(inference_seconds),
        "evaluation":
            "threshold_sensitivity_0.95",
        "feature_representation":
            "shared precomputed TF-IDF matrices",
    })

    pd.DataFrame({
        "external_final_index":
            ext["external_final_index"],
        "archive_path":
            ext["archive_path"],
        "raw_sha256":
            ext["raw_sha256"],
        "normalized_sha256":
            ext["normalized_sha256"],
        "y_true":
            y_true,
        "y_pred":
            pred,
    }).to_csv(
        out / "external_predictions.csv",
        index=False,
    )

    (
        out / "external_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ) + "\n"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        ),
        flush=True,
    )

    return metrics


def main():

    train = pd.read_csv(
        TRAIN_FILE
    )

    ext = pd.read_csv(
        EXTERNAL_FILE
    )

    assert len(train) == 4322
    assert len(ext) == 19683

    y_train = (
        train["label"].to_numpy()
    )

    y_ext = (
        ext["label"].to_numpy()
    )

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

    ext_texts = (
        load_external_texts(ext)
    )

    # ---------------------------------------------
    # CHAR representation -- computed exactly once
    # ---------------------------------------------

    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200000,
    )

    print(
        "\n=== FIT CHAR TF-IDF ===",
        flush=True,
    )

    t0 = time.perf_counter()

    Xtr_char = char_vec.fit_transform(
        train_texts
    )

    char_fit_s = (
        time.perf_counter() - t0
    )

    print(
        "Train char shape:",
        Xtr_char.shape,
        flush=True,
    )

    print(
        "Char fit seconds:",
        char_fit_s,
        flush=True,
    )

    print(
        "\n=== TRANSFORM EXTERNAL CHAR ===",
        flush=True,
    )

    t0 = time.perf_counter()

    Xext_char = char_vec.transform(
        ext_texts
    )

    char_transform_s = (
        time.perf_counter() - t0
    )

    print(
        "External char shape:",
        Xext_char.shape,
        flush=True,
    )

    print(
        "Char external transform seconds:",
        char_transform_s,
        flush=True,
    )

    # ---------------------------------------------
    # WORD representation -- computed exactly once
    # ---------------------------------------------

    word_vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=(
            r"(?u)\b[\w\-$.:\\/]+\b"
        ),
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
    )

    print(
        "\n=== FIT WORD TF-IDF ===",
        flush=True,
    )

    t0 = time.perf_counter()

    Xtr_word = word_vec.fit_transform(
        train_texts
    )

    word_fit_s = (
        time.perf_counter() - t0
    )

    print(
        "Train word shape:",
        Xtr_word.shape,
        flush=True,
    )

    print(
        "Word fit seconds:",
        word_fit_s,
        flush=True,
    )

    print(
        "\n=== TRANSFORM EXTERNAL WORD ===",
        flush=True,
    )

    t0 = time.perf_counter()

    Xext_word = word_vec.transform(
        ext_texts
    )

    word_transform_s = (
        time.perf_counter() - t0
    )

    print(
        "External word shape:",
        Xext_word.shape,
        flush=True,
    )

    print(
        "Word external transform seconds:",
        word_transform_s,
        flush=True,
    )

    # ---------------------------------------------
    # Hybrid char + word
    # FeatureUnion is equivalent to sparse hstack
    # here because no transformer weights are used.
    # ---------------------------------------------

    Xtr_cw = hstack(
        [Xtr_char, Xtr_word],
        format="csr",
    )

    Xext_cw = hstack(
        [Xext_char, Xext_word],
        format="csr",
    )

    summary = []

    # ---------------------------------------------
    # LR char+word
    # ---------------------------------------------

    print(
        "\n=== lr_char_word ===",
        flush=True,
    )

    clf = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    clf.fit(Xtr_cw, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = clf.predict(Xext_cw)
    pred_s = time.perf_counter() - t0

    summary.append(
        save_result(
            "lr_char_word",
            2.0,
            ext,
            y_ext,
            pred,
            train_s,
            pred_s,
        )
    )

    # ---------------------------------------------
    # LR char
    # ---------------------------------------------

    print(
        "\n=== lr_char ===",
        flush=True,
    )

    clf = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    clf.fit(Xtr_char, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = clf.predict(Xext_char)
    pred_s = time.perf_counter() - t0

    summary.append(
        save_result(
            "lr_char",
            2.0,
            ext,
            y_ext,
            pred,
            train_s,
            pred_s,
        )
    )

    # ---------------------------------------------
    # SVM char
    # ---------------------------------------------

    print(
        "\n=== svm_char ===",
        flush=True,
    )

    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",
    )

    t0 = time.perf_counter()
    clf.fit(Xtr_char, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = clf.predict(Xext_char)
    pred_s = time.perf_counter() - t0

    summary.append(
        save_result(
            "svm_char",
            1.0,
            ext,
            y_ext,
            pred,
            train_s,
            pred_s,
        )
    )

    # ---------------------------------------------
    # SVM word
    # ---------------------------------------------

    print(
        "\n=== svm_word ===",
        flush=True,
    )

    clf = LinearSVC(
        C=2.0,
        class_weight="balanced",
    )

    t0 = time.perf_counter()
    clf.fit(Xtr_word, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = clf.predict(Xext_word)
    pred_s = time.perf_counter() - t0

    summary.append(
        save_result(
            "svm_word",
            2.0,
            ext,
            y_ext,
            pred,
            train_s,
            pred_s,
        )
    )

    pd.DataFrame(
        summary
    ).to_csv(
        OUT_ROOT /
        "external_baseline_summary.csv",
        index=False,
    )

    timing = {
        "char_fit_seconds":
            char_fit_s,
        "char_external_transform_seconds":
            char_transform_s,
        "word_fit_seconds":
            word_fit_s,
        "word_external_transform_seconds":
            word_transform_s,
    }

    (
        OUT_ROOT /
        "feature_timing.json"
    ).write_text(
        json.dumps(
            timing,
            indent=2,
        ) + "\n"
    )

    print(
        "\n=== FEATURE TIMING ===",
        flush=True,
    )

    print(
        json.dumps(
            timing,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\n=== DONE ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
