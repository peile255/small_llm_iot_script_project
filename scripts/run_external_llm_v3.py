#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

import pandas as pd


CONFIG = Path(
    "configs/ollama_external_v3.yaml"
)

MANIFEST = Path(
    "outputs/paper_v3/"
    "frozen_external_llm_subset_v1/"
    "external_llm_subset_2000.csv"
)

ALLOWED_MODELS = {
    "mistral_7b",
    "phi4",
    "gemma3_27b",
    "qwq_32b",
    "deepseek_r1_32b",
}


def validate_subset():
    df = pd.read_csv(MANIFEST)

    assert len(df) == 2000, (
        f"Expected 2000 rows, got {len(df)}"
    )

    assert (df["label"] == 0).sum() == 1000
    assert (df["label"] == 1).sum() == 1000

    assert (
        df["normalized_sha256"].nunique()
        == 2000
    )

    assert (
        df["nearest_primary_similarity"].max()
        < 0.90
    )

    missing = [
        p
        for p in df["path"]
        if not Path(p).exists()
    ]

    assert not missing, (
        f"Missing local scripts: {len(missing)}"
    )

    print("=== EXTERNAL SUBSET VALIDATED ===")
    print("Rows:", len(df))
    print(
        "Label counts:",
        df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    print(
        "Max primary similarity:",
        df[
            "nearest_primary_similarity"
        ].max()
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        required=True,
        choices=sorted(ALLOWED_MODELS),
    )

    args = ap.parse_args()

    validate_subset()

    output_root = Path(
        os.environ.get(
            "OUTPUT_ROOT",
            "outputs/paper_v3/external_llm",
        )
    )

    model_dir = (
        output_root / args.model
    )

    print("\n=== EXTERNAL LLM RUN ===")
    print("Model:", args.model)
    print("Config:", CONFIG)
    print("Output:", model_dir)

    cmd = [
        sys.executable,
        "run_ollama.py",
        "--config",
        str(CONFIG),
        "--model",
        args.model,
    ]

    subprocess.run(
        cmd,
        check=True,
    )

    pred = (
        model_dir /
        "test_predictions.csv"
    )

    metrics = (
        model_dir /
        "metrics.json"
    )

    if not pred.exists():
        raise RuntimeError(
            f"Missing predictions: {pred}"
        )

    if not metrics.exists():
        raise RuntimeError(
            f"Missing metrics: {metrics}"
        )

    pdf = pd.read_csv(pred)

    if len(pdf) != 2000:
        raise RuntimeError(
            f"Expected 2000 predictions, "
            f"got {len(pdf)}"
        )

    print(
        "\n=== OUTPUT VALIDATION PASSED ==="
    )

    print(
        "Predictions:",
        len(pdf),
    )

    print(
        "Metrics:",
        json.loads(
            metrics.read_text()
        )
    )


if __name__ == "__main__":
    main()
