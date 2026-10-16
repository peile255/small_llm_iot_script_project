from pathlib import Path
import json

import pandas as pd


OLD_ROOT = Path(
    "outputs/paper_v3/external_baselines_v3_1"
)

NEW_ROOT = Path(
    "outputs/paper_v4/threshold_sensitivity_v1/"
    "threshold_095_predictions"
)

OUT = Path(
    "outputs/paper_v4/threshold_sensitivity_v1/"
    "threshold_095_identity_check.json"
)

MODELS = [
    "lr_char_word",
    "lr_char",
    "svm_char",
    "svm_word",
]

summary = {}
all_exact = True

for model in MODELS:

    old = pd.read_csv(
        OLD_ROOT / model / "external_predictions.csv"
    )

    new = pd.read_csv(
        NEW_ROOT / model / "external_predictions.csv"
    )

    merged = old[
        [
            "normalized_sha256",
            "y_true",
            "y_pred",
        ]
    ].merge(
        new[
            [
                "normalized_sha256",
                "y_true",
                "y_pred",
            ]
        ],
        on="normalized_sha256",
        suffixes=("_old", "_new"),
        how="inner",
        validate="one_to_one",
    )

    n_match = len(merged)

    label_equal = (
        merged["y_true_old"]
        == merged["y_true_new"]
    )

    pred_equal = (
        merged["y_pred_old"]
        == merged["y_pred_new"]
    )

    prediction_mismatches = int(
        (~pred_equal).sum()
    )

    label_mismatches = int(
        (~label_equal).sum()
    )

    exact = (
        n_match == 19632
        and label_mismatches == 0
        and prediction_mismatches == 0
    )

    all_exact &= exact

    summary[model] = {
        "matched_rows": int(n_match),
        "label_mismatches": label_mismatches,
        "prediction_mismatches": prediction_mismatches,
        "exact_identity": bool(exact),
    }

summary["all_models_exact_identity"] = bool(
    all_exact
)

OUT.write_text(
    json.dumps(summary, indent=2) + "\n"
)

print(json.dumps(summary, indent=2))

if not all_exact:
    raise SystemExit(
        "IDENTITY CHECK FAILED — "
        "do not use threshold 0.95 results."
    )

print(
    "\nIDENTITY CHECK PASSED: "
    "all 19,632 frozen 0.90 predictions "
    "were reproduced exactly."
)
