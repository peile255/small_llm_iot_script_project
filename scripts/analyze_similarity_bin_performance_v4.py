from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import (
    ks_2samp,
    spearmanr,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)


ROOT = Path(".")

SIM_DIR = (
    ROOT
    / "outputs/paper_v4/distribution_shift_v1"
)

STATS_DIR = (
    ROOT
    / "outputs/paper_v3/"
      "frozen_external_results_v1/"
      "external_statistics_v3"
)

PAIRED_CSV = (
    STATS_DIR
    / "paired_external_2000_predictions.csv"
)

EXT_SIM_CSV = (
    SIM_DIR
    / "external_2000_train_similarity.csv"
)

INT_SIM_CSV = (
    SIM_DIR
    / "internal_test_train_similarity.csv"
)

OUT = (
    ROOT
    / "outputs/paper_v4/"
      "distribution_shift_v1/"
      "performance_by_similarity"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_COLS = {
    "LR Char+Word": "pred_lr_char_word",
    "LR Char": "pred_lr_char",
    "SVM Char": "pred_svm_char",
    "SVM Word": "pred_svm_word",
    "Mistral-7B": "pred_mistral_7b",
    "Phi-4": "pred_phi4",
    "Gemma-3-27B": "pred_gemma3_27b",
    "QwQ-32B": "pred_qwq_32b",
    "DeepSeek-R1-32B":
        "pred_deepseek_r1_32b",
}

PRIMARY = "LR Char+Word"


def safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b


def confusion(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tp = int(
        np.sum((y == 1) & (pred == 1))
    )
    tn = int(
        np.sum((y == 0) & (pred == 0))
    )
    fp = int(
        np.sum((y == 0) & (pred == 1))
    )
    fn = int(
        np.sum((y == 1) & (pred == 0))
    )

    return tp, tn, fp, fn


def full_metrics(df, pred_col):
    y = df["y_true"].to_numpy()
    p = df[pred_col].to_numpy()

    tp, tn, fp, fn = confusion(y, p)

    return {
        "n": len(df),
        "n_benign": int(
            np.sum(y == 0)
        ),
        "n_malicious": int(
            np.sum(y == 1)
        ),
        "prevalence": float(
            np.mean(y)
        ),
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy":
            balanced_accuracy_score(y, p),
        "precision": precision_score(
            y,
            p,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            p,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            p,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(y, p),
        "fpr": safe_div(
            fp,
            fp + tn,
        ),
        "fnr": safe_div(
            fn,
            fn + tp,
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# --------------------------------------------------
# Load + exact alignment
# --------------------------------------------------

print("Loading data...")

paired = pd.read_csv(PAIRED_CSV)
ext = pd.read_csv(EXT_SIM_CSV)
internal = pd.read_csv(INT_SIM_CSV)

print(
    "paired:",
    paired.shape,
)
print(
    "external similarity:",
    ext.shape,
)
print(
    "internal similarity:",
    internal.shape,
)

assert len(paired) == 2000
assert len(ext) == 2000
assert len(internal) == 1265


merged = paired.merge(
    ext[
        [
            "external_llm_index",
            "external_final_index",
            "label",
            "nearest_train_similarity",
        ]
    ],
    on=[
        "external_llm_index",
        "external_final_index",
    ],
    how="inner",
    validate="one_to_one",
)

assert len(merged) == 2000

assert np.array_equal(
    merged["y_true"].to_numpy(),
    merged["label"].to_numpy(),
)

merged = merged.drop(
    columns=["label"]
)

print(
    "Merged rows:",
    len(merged),
)


# --------------------------------------------------
# 1. Label-stratified internal/external distance
# --------------------------------------------------

distance_rows = []
distance_tests = []

for label, label_name in [
    (0, "benign"),
    (1, "malicious"),
]:
    int_x = (
        internal.loc[
            internal["label"] == label,
            "nearest_train_similarity",
        ]
        .to_numpy()
    )

    ext_x = (
        merged.loc[
            merged["y_true"] == label,
            "nearest_train_similarity",
        ]
        .to_numpy()
    )

    for dist_name, x in [
        ("internal_test", int_x),
        ("external_paired", ext_x),
    ]:
        distance_rows.append(
            {
                "label": label,
                "class": label_name,
                "distribution": dist_name,
                "n": len(x),
                "mean": np.mean(x),
                "std": np.std(
                    x,
                    ddof=1,
                ),
                "q25": np.quantile(
                    x,
                    0.25,
                ),
                "median": np.median(x),
                "q75": np.quantile(
                    x,
                    0.75,
                ),
                "min": np.min(x),
                "max": np.max(x),
            }
        )

    ks = ks_2samp(
        int_x,
        ext_x,
        alternative="two-sided",
        method="auto",
    )

    distance_tests.append(
        {
            "label": label,
            "class": label_name,
            "internal_n":
                len(int_x),
            "external_n":
                len(ext_x),
            "ks_D":
                ks.statistic,
            "ks_p":
                ks.pvalue,
            "mean_difference_external_minus_internal":
                np.mean(ext_x)
                - np.mean(int_x),
            "median_difference_external_minus_internal":
                np.median(ext_x)
                - np.median(int_x),
        }
    )


distance_df = pd.DataFrame(
    distance_rows
)

distance_test_df = pd.DataFrame(
    distance_tests
)

distance_df.to_csv(
    OUT
    / "distance_summary_by_class.csv",
    index=False,
)

distance_test_df.to_csv(
    OUT
    / "distance_ks_by_class.csv",
    index=False,
)


# --------------------------------------------------
# 2. Overall external quintiles
#
# Descriptive only because prevalence changes heavily.
# --------------------------------------------------

merged[
    "overall_similarity_quintile"
] = pd.qcut(
    merged[
        "nearest_train_similarity"
    ],
    q=5,
    labels=[
        "Q1_farthest",
        "Q2",
        "Q3",
        "Q4",
        "Q5_closest",
    ],
)


overall_rows = []

for q, g in merged.groupby(
    "overall_similarity_quintile",
    observed=False,
):
    for model, col in MODEL_COLS.items():
        m = full_metrics(
            g,
            col,
        )

        overall_rows.append(
            {
                "similarity_quintile":
                    str(q),
                "similarity_min":
                    g[
                        "nearest_train_similarity"
                    ].min(),
                "similarity_mean":
                    g[
                        "nearest_train_similarity"
                    ].mean(),
                "similarity_max":
                    g[
                        "nearest_train_similarity"
                    ].max(),
                "model": model,
                **m,
            }
        )


overall_df = pd.DataFrame(
    overall_rows
)

overall_df.to_csv(
    OUT
    / "overall_quintile_metrics_descriptive.csv",
    index=False,
)


# --------------------------------------------------
# 3. LABEL-STRATIFIED quintiles
#
# Critical analysis:
# benign similarity quintiles -> FPR
# malicious similarity quintiles -> recall/FNR
# --------------------------------------------------

merged[
    "class_similarity_quintile"
] = None


for label in [0, 1]:
    mask = (
        merged["y_true"] == label
    )

    q = pd.qcut(
        merged.loc[
            mask,
            "nearest_train_similarity",
        ],
        q=5,
        labels=[
            "Q1_farthest",
            "Q2",
            "Q3",
            "Q4",
            "Q5_closest",
        ],
    )

    merged.loc[
        mask,
        "class_similarity_quintile",
    ] = q.astype(str).values


class_rows = []

for label, class_name in [
    (0, "benign"),
    (1, "malicious"),
]:
    subset = merged[
        merged["y_true"] == label
    ].copy()

    for q, g in subset.groupby(
        "class_similarity_quintile",
        sort=False,
    ):
        for model, col in MODEL_COLS.items():
            p = g[col].to_numpy()

            if label == 0:
                # all samples benign:
                # error rate = FPR
                errors = int(
                    np.sum(p == 1)
                )

                class_metric = (
                    1
                    - errors / len(g)
                )

                error_rate = (
                    errors / len(g)
                )

                metric_name = (
                    "specificity"
                )

            else:
                # all samples malicious:
                # correct rate = recall
                errors = int(
                    np.sum(p == 0)
                )

                class_metric = (
                    1
                    - errors / len(g)
                )

                error_rate = (
                    errors / len(g)
                )

                metric_name = "recall"

            class_rows.append(
                {
                    "class":
                        class_name,
                    "label":
                        label,
                    "similarity_quintile":
                        q,
                    "n":
                        len(g),
                    "similarity_min":
                        g[
                            "nearest_train_similarity"
                        ].min(),
                    "similarity_mean":
                        g[
                            "nearest_train_similarity"
                        ].mean(),
                    "similarity_median":
                        g[
                            "nearest_train_similarity"
                        ].median(),
                    "similarity_max":
                        g[
                            "nearest_train_similarity"
                        ].max(),
                    "model":
                        model,
                    "metric":
                        metric_name,
                    "correct_rate":
                        class_metric,
                    "error_rate":
                        error_rate,
                    "n_errors":
                        errors,
                }
            )


class_df = pd.DataFrame(
    class_rows
)

class_df.to_csv(
    OUT
    / "class_stratified_quintile_performance.csv",
    index=False,
)


# --------------------------------------------------
# 4. Primary-vs-model paired error differences,
#    within class-specific quintiles.
#
# Positive net_correct_gain means model fixes more
# primary errors than it introduces.
# --------------------------------------------------

gain_rows = []

primary_col = MODEL_COLS[
    PRIMARY
]

for label, class_name in [
    (0, "benign"),
    (1, "malicious"),
]:

    subset = merged[
        merged["y_true"] == label
    ].copy()

    for q, g in subset.groupby(
        "class_similarity_quintile",
        sort=False,
    ):
        y = g[
            "y_true"
        ].to_numpy()

        primary_correct = (
            g[primary_col].to_numpy()
            == y
        )

        for model, col in MODEL_COLS.items():

            if model == PRIMARY:
                continue

            model_correct = (
                g[col].to_numpy()
                == y
            )

            # model correct, primary wrong
            fixes = int(
                np.sum(
                    model_correct
                    & ~primary_correct
                )
            )

            # primary correct, model wrong
            harms = int(
                np.sum(
                    primary_correct
                    & ~model_correct
                )
            )

            gain_rows.append(
                {
                    "class":
                        class_name,
                    "label":
                        label,
                    "similarity_quintile":
                        q,
                    "model":
                        model,
                    "n":
                        len(g),
                    "fixes_primary_errors":
                        fixes,
                    "introduces_errors":
                        harms,
                    "net_correct_gain":
                        fixes - harms,
                    "net_correct_gain_rate":
                        (
                            fixes - harms
                        )
                        / len(g),
                }
            )


gain_df = pd.DataFrame(
    gain_rows
)

gain_df.to_csv(
    OUT
    / "paired_net_correct_gain_by_class_quintile.csv",
    index=False,
)


# --------------------------------------------------
# 5. Continuous association:
# similarity vs binary error,
# separately within benign/malicious.
#
# Spearman is descriptive, not causal.
# --------------------------------------------------

corr_rows = []

for label, class_name in [
    (0, "benign"),
    (1, "malicious"),
]:

    g = merged[
        merged["y_true"] == label
    ]

    sim = g[
        "nearest_train_similarity"
    ].to_numpy()

    for model, col in MODEL_COLS.items():

        error = (
            g[col].to_numpy()
            != label
        ).astype(int)

        if np.all(
            error == error[0]
        ):
            rho = np.nan
            pval = np.nan
        else:
            result = spearmanr(
                sim,
                error,
            )

            rho = (
                result.statistic
            )
            pval = (
                result.pvalue
            )

        corr_rows.append(
            {
                "class":
                    class_name,
                "label":
                    label,
                "model":
                    model,
                "n":
                    len(g),
                "error_count":
                    int(error.sum()),
                "spearman_rho_similarity_vs_error":
                    rho,
                "p_value":
                    pval,
            }
        )


corr_df = pd.DataFrame(
    corr_rows
)

corr_df.to_csv(
    OUT
    / "similarity_error_spearman_by_class.csv",
    index=False,
)


# --------------------------------------------------
# 6. Save fully merged dataset
# --------------------------------------------------

merged.to_csv(
    OUT
    / "paired_external_2000_with_train_similarity.csv",
    index=False,
)


# --------------------------------------------------
# Summary JSON
# --------------------------------------------------

summary = {
    "analysis_warning": (
        "Overall similarity quintiles are strongly "
        "confounded with class prevalence. "
        "Primary RQ6 interpretation should use "
        "label-stratified quintiles."
    ),
    "n_external": int(
        len(merged)
    ),
    "n_benign": int(
        (merged["y_true"] == 0).sum()
    ),
    "n_malicious": int(
        (merged["y_true"] == 1).sum()
    ),
    "models": list(
        MODEL_COLS.keys()
    ),
}


with open(
    OUT / "analysis_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# --------------------------------------------------
# Console output
# --------------------------------------------------

print("\n=== DISTANCE BY CLASS ===")
print(
    distance_df.to_string(
        index=False
    )
)

print("\n=== KS BY CLASS ===")
print(
    distance_test_df.to_string(
        index=False
    )
)

print(
    "\n=== CLASS-STRATIFIED "
    "QUINTILE PERFORMANCE ==="
)

for cls in [
    "benign",
    "malicious",
]:
    print(
        f"\n--- {cls.upper()} ---"
    )

    x = class_df[
        class_df["class"] == cls
    ][
        [
            "similarity_quintile",
            "model",
            "correct_rate",
            "error_rate",
            "n_errors",
        ]
    ]

    print(
        x.to_string(
            index=False
        )
    )


print(
    "\n=== PAIRED NET CORRECT "
    "GAIN VS PRIMARY ==="
)

for cls in [
    "benign",
    "malicious",
]:
    print(
        f"\n--- {cls.upper()} ---"
    )

    x = gain_df[
        gain_df["class"] == cls
    ]

    print(
        x.to_string(
            index=False
        )
    )


print(
    "\n=== SIMILARITY VS ERROR "
    "CORRELATION ==="
)

print(
    corr_df.to_string(
        index=False
    )
)


print("\nOutputs:")
for p in sorted(
    OUT.iterdir()
):
    print(" ", p)
