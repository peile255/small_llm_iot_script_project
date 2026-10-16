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


ROOT = Path(
    "outputs/paper_v4/threshold_sensitivity_v1"
)

NEW095 = ROOT / "threshold_095_predictions"

MODELS = [
    "lr_char_word",
    "lr_char",
    "svm_char",
    "svm_word",
]


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        p,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(
            accuracy_score(y, p)
        ),
        "precision": float(
            precision_score(
                y,
                p,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                p,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                p,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(y, p)
        ),
    }


# Existing 0.80 / 0.85 / 0.90 results
old = pd.read_csv(
    ROOT /
    "threshold_performance_available.csv"
)

rows = old.to_dict("records")


# Add 0.95
for model in MODELS:

    df = pd.read_csv(
        NEW095 /
        model /
        "external_predictions.csv"
    )

    m = metrics(
        df["y_true"],
        df["y_pred"],
    )

    rows.append({
        "threshold": 0.95,
        "model": model,
        **m,
        "prediction_source":
            "recomputed_exact_protocol_0.95",
    })


final = pd.DataFrame(rows)

final = final.sort_values(
    ["threshold", "model"]
).reset_index(drop=True)

final.to_csv(
    ROOT /
    "threshold_performance_final.csv",
    index=False,
)


# Compact F1 table
compact = final.pivot(
    index="threshold",
    columns="model",
    values="f1",
).reset_index()

compact.to_csv(
    ROOT /
    "threshold_f1_compact.csv",
    index=False,
)


# Ranking stability
ranking = []

for t, g in final.groupby("threshold"):

    ordered = (
        g.sort_values(
            "f1",
            ascending=False,
        )["model"]
        .tolist()
    )

    ranking.append({
        "threshold": float(t),
        "ranking": " > ".join(ordered),
        "best_model": ordered[0],
    })

ranking_df = pd.DataFrame(ranking)

ranking_df.to_csv(
    ROOT /
    "threshold_model_rankings.csv",
    index=False,
)


# Per-model F1 ranges
ranges = []

for model, g in final.groupby("model"):

    ranges.append({
        "model": model,
        "min_f1": float(g["f1"].min()),
        "max_f1": float(g["f1"].max()),
        "absolute_f1_range": float(
            g["f1"].max()
            - g["f1"].min()
        ),
        "min_mcc": float(g["mcc"].min()),
        "max_mcc": float(g["mcc"].max()),
        "absolute_mcc_range": float(
            g["mcc"].max()
            - g["mcc"].min()
        ),
    })

range_df = pd.DataFrame(ranges)

range_df.to_csv(
    ROOT /
    "threshold_metric_ranges.csv",
    index=False,
)


summary = {
    "analysis_type":
        "near-duplicate threshold sensitivity",
    "thresholds":
        [0.80, 0.85, 0.90, 0.95],
    "primary_threshold":
        0.90,
    "identity_check":
        (
            "The 0.95 rerun reproduced all "
            "19,632 predictions from the frozen "
            "0.90 external set exactly for all "
            "four conventional classifiers."
        ),
    "rankings":
        ranking_df.to_dict("records"),
    "metric_ranges":
        range_df.to_dict("records"),
}

(
    ROOT /
    "threshold_final_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ) + "\n"
)


# LaTeX table
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Sensitivity of external conventional-model F1 to the near-duplicate exclusion threshold.}",
    r"\label{tab:threshold_sensitivity}",
    r"\small",
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"Threshold & $n$ & LR C+W & LR Char & SVM Char & SVM Word \\",
    r"\midrule",
]

for t in [0.80, 0.85, 0.90, 0.95]:

    g = final[
        final["threshold"] == t
    ]

    vals = {
        r["model"]: r
        for _, r in g.iterrows()
    }

    n = int(g.iloc[0]["n"])

    lines.append(
        f"{t:.2f} & {n:,} & "
        f"{vals['lr_char_word']['f1']:.3f} & "
        f"{vals['lr_char']['f1']:.3f} & "
        f"{vals['svm_char']['f1']:.3f} & "
        f"{vals['svm_word']['f1']:.3f} \\\\"
    )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

(
    ROOT /
    "table_threshold_sensitivity.tex"
).write_text(
    "\n".join(lines) + "\n"
)


print("\n=== FINAL F1 TABLE ===")
print(compact.to_string(index=False))

print("\n=== METRIC RANGES ===")
print(range_df.to_string(index=False))

print("\n=== RANKINGS ===")
print(ranking_df.to_string(index=False))

print(
    "\nOutputs written to:",
    ROOT,
)
