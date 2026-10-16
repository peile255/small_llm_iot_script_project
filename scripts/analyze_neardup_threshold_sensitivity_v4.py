from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

NEAREST_FILE = Path(
    "outputs/paper_v3/external_neardup/"
    "external_nearest_primary.csv"
)

PRED_ROOT = Path(
    "outputs/paper_v3/external_baselines_v3_1"
)

OUTDIR = Path(
    "outputs/paper_v4/"
    "threshold_sensitivity_v1"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = {
    "lr_char_word":
        PRED_ROOT /
        "lr_char_word/external_predictions.csv",

    "lr_char":
        PRED_ROOT /
        "lr_char/external_predictions.csv",

    "svm_char":
        PRED_ROOT /
        "svm_char/external_predictions.csv",

    "svm_word":
        PRED_ROOT /
        "svm_word/external_predictions.csv",
}


def metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y,
                pred,
            )
        ),
    }


def main():
    nearest = pd.read_csv(
        NEAREST_FILE
    )

    print(
        "exact-clean rows:",
        len(nearest),
    )

    print(
        "columns:",
        list(nearest.columns),
    )

    # --------------------------------
    # Build retained manifests
    # --------------------------------

    threshold_summary = []

    retained_sets = {}

    for t in THRESHOLDS:
        retained = nearest[
            nearest[
                "nearest_primary_similarity"
            ] < t
        ].copy()

        removed = nearest[
            nearest[
                "nearest_primary_similarity"
            ] >= t
        ].copy()

        retained_sets[t] = retained

        row = {
            "threshold": t,
            "retained_n":
                int(len(retained)),
            "removed_n":
                int(len(removed)),
            "retained_label_0":
                int(
                    (retained["label"] == 0)
                    .sum()
                ),
            "retained_label_1":
                int(
                    (retained["label"] == 1)
                    .sum()
                ),
            "removed_label_0":
                int(
                    (removed["label"] == 0)
                    .sum()
                ),
            "removed_label_1":
                int(
                    (removed["label"] == 1)
                    .sum()
                ),
        }

        threshold_summary.append(row)

        retained.to_csv(
            OUTDIR /
            f"retained_threshold_{t:.2f}.csv",
            index=False,
        )

        removed.to_csv(
            OUTDIR /
            f"removed_threshold_{t:.2f}.csv",
            index=False,
        )

    threshold_summary_df = pd.DataFrame(
        threshold_summary
    )

    threshold_summary_df.to_csv(
        OUTDIR /
        "threshold_sample_counts.csv",
        index=False,
    )

    print(
        "\n=== THRESHOLD COUNTS ==="
    )
    print(
        threshold_summary_df.to_string(
            index=False
        )
    )

    # --------------------------------
    # Determine samples newly included
    # at 0.95 relative to 0.90
    # --------------------------------

    extra95 = nearest[
        (
            nearest[
                "nearest_primary_similarity"
            ] >= 0.90
        )
        &
        (
            nearest[
                "nearest_primary_similarity"
            ] < 0.95
        )
    ].copy()

    extra95.to_csv(
        OUTDIR /
        "extra_090_to_095.csv",
        index=False,
    )

    print(
        "\n0.90 <= similarity < 0.95:",
        len(extra95),
    )

    print(
        "labels:",
        extra95[
            "label"
        ].value_counts()
        .sort_index()
        .to_dict(),
    )

    # --------------------------------
    # Evaluate thresholds <= 0.90
    # using existing predictions
    # --------------------------------

    rows = []

    prediction_cache = {}

    for model, path in MODELS.items():
        pred = pd.read_csv(path)

        if pred[
            "normalized_sha256"
        ].duplicated().any():
            raise RuntimeError(
                f"Duplicate SHA in {model}"
            )

        prediction_cache[
            model
        ] = pred

    for t in [
        0.80,
        0.85,
        0.90,
    ]:
        retained = retained_sets[t]

        keep = retained[
            [
                "normalized_sha256",
                "label",
            ]
        ].copy()

        for model, pred in (
            prediction_cache.items()
        ):
            merged = keep.merge(
                pred[
                    [
                        "normalized_sha256",
                        "y_true",
                        "y_pred",
                    ]
                ],
                on="normalized_sha256",
                how="inner",
                validate="one_to_one",
            )

            if (
                len(merged)
                != len(retained)
            ):
                raise RuntimeError(
                    f"{model} threshold {t}: "
                    f"matched {len(merged)} "
                    f"of {len(retained)}"
                )

            if not np.array_equal(
                merged[
                    "label"
                ].astype(int)
                .to_numpy(),
                merged[
                    "y_true"
                ].astype(int)
                .to_numpy(),
            ):
                raise RuntimeError(
                    f"Label mismatch: "
                    f"{model}, threshold={t}"
                )

            m = metrics(
                merged["y_true"],
                merged["y_pred"],
            )

            rows.append(
                {
                    "threshold": t,
                    "model": model,
                    **m,
                    "prediction_source":
                        "existing_0.90_predictions",
                }
            )

    perf = pd.DataFrame(rows)

    perf.to_csv(
        OUTDIR /
        "threshold_performance_available.csv",
        index=False,
    )

    print(
        "\n=== PERFORMANCE "
        "(0.80 / 0.85 / 0.90) ==="
    )

    display_cols = [
        "threshold",
        "model",
        "n",
        "precision",
        "recall",
        "f1",
        "mcc",
    ]

    print(
        perf[
            display_cols
        ].to_string(
            index=False
        )
    )

    # --------------------------------
    # Summary JSON
    # --------------------------------

    summary = {
        "analysis_type":
            "near-duplicate threshold sensitivity",
        "similarity_definition":
            "nearest frozen-primary char_wb TF-IDF cosine similarity",
        "threshold_rule":
            "retain if nearest_primary_similarity < threshold",
        "thresholds":
            THRESHOLDS,
        "exact_clean_n":
            int(len(nearest)),
        "threshold_counts":
            threshold_summary,
        "extra_090_to_095_n":
            int(len(extra95)),
        "extra_090_to_095_labels":
            {
                str(k): int(v)
                for k, v in
                extra95[
                    "label"
                ].value_counts()
                .sort_index()
                .to_dict()
                .items()
            },
        "note":
            (
                "Performance for thresholds "
                "0.80, 0.85, and 0.90 is "
                "computed by filtering the "
                "existing 0.90 prediction set. "
                "Threshold 0.95 additionally "
                "requires predictions for the "
                "samples with similarity in "
                "[0.90, 0.95)."
            ),
    }

    with open(
        OUTDIR /
        "threshold_sensitivity_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\nOutputs:",
        OUTDIR,
    )


if __name__ == "__main__":
    main()
