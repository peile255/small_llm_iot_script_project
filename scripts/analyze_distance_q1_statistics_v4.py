from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import binomtest


ROOT = Path(".")

INPUT = (
    ROOT
    / "outputs/paper_v4/"
      "distribution_shift_v1/"
      "performance_by_similarity/"
      "paired_external_2000_with_train_similarity.csv"
)

OUT = (
    ROOT
    / "outputs/paper_v4/"
      "distribution_shift_v1/"
      "q1_statistics"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


PRIMARY_NAME = "LR Char+Word"
PRIMARY_COL = "pred_lr_char_word"

CONVENTIONAL = {
    "LR Char":
        "pred_lr_char",
    "SVM Char":
        "pred_svm_char",
    "SVM Word":
        "pred_svm_word",
}

LLMS = {
    "Mistral-7B":
        "pred_mistral_7b",
    "Phi-4":
        "pred_phi4",
    "Gemma-3-27B":
        "pred_gemma3_27b",
    "QwQ-32B":
        "pred_qwq_32b",
    "DeepSeek-R1-32B":
        "pred_deepseek_r1_32b",
}

ALL_MODELS = {
    **CONVENTIONAL,
    **LLMS,
}

N_BOOT = 10000
SEED = 20260921


def holm_adjust(pvalues):
    """
    Holm step-down adjusted p-values.
    Returns adjusted values in original order.
    """
    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(pvalues)

    order = np.argsort(
        pvalues
    )

    sorted_p = pvalues[
        order
    ]

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, p in enumerate(
        sorted_p
    ):
        value = (
            (m - i) * p
        )

        value = min(
            1.0,
            value,
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted_sorted[i] = (
            running_max
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted


def exact_mcnemar(
    primary_correct,
    model_correct,
):
    """
    b = primary correct, model wrong
    c = primary wrong, model correct
    """
    primary_correct = np.asarray(
        primary_correct,
        dtype=bool,
    )

    model_correct = np.asarray(
        model_correct,
        dtype=bool,
    )

    b = int(
        np.sum(
            primary_correct
            & ~model_correct
        )
    )

    c = int(
        np.sum(
            ~primary_correct
            & model_correct
        )
    )

    n_disc = b + c

    if n_disc == 0:
        p = 1.0

    else:
        p = binomtest(
            k=min(b, c),
            n=n_disc,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return b, c, n_disc, p


def paired_bootstrap_delta(
    primary_pred,
    model_pred,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    All samples are malicious (y=1),
    so recall is simply mean(prediction).
    Delta recall = model recall - primary recall.
    """

    primary_pred = np.asarray(
        primary_pred,
        dtype=np.float64,
    )

    model_pred = np.asarray(
        model_pred,
        dtype=np.float64,
    )

    assert (
        len(primary_pred)
        == len(model_pred)
    )

    n = len(primary_pred)

    rng = np.random.default_rng(
        seed
    )

    deltas = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(
        n_boot
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        primary_recall = (
            primary_pred[idx]
            .mean()
        )

        model_recall = (
            model_pred[idx]
            .mean()
        )

        deltas[i] = (
            model_recall
            - primary_recall
        )

    lo, hi = np.quantile(
        deltas,
        [0.025, 0.975],
    )

    return (
        float(lo),
        float(hi),
    )


print(
    "Loading:",
    INPUT,
)

df = pd.read_csv(
    INPUT
)

print(
    "shape:",
    df.shape,
)


required = [
    "y_true",
    "nearest_train_similarity",
    "class_similarity_quintile",
    PRIMARY_COL,
    *ALL_MODELS.values(),
]

for c in required:
    assert c in df.columns, (
        f"Missing required column: {c}"
    )


# ---------------------------------------
# Exact same Q1-farthest malicious stratum
# produced by previous class-stratified
# similarity analysis.
# ---------------------------------------

q1 = df[
    (df["y_true"] == 1)
    &
    (
        df[
            "class_similarity_quintile"
        ]
        == "Q1_farthest"
    )
].copy()


print(
    "Q1 malicious n:",
    len(q1),
)

assert len(q1) == 200

assert q1[
    "y_true"
].eq(1).all()


# ---------------------------------------
# Save Q1 sample manifest
# ---------------------------------------

q1.to_csv(
    OUT
    / "q1_farthest_malicious_200.csv",
    index=False,
)


primary = q1[
    PRIMARY_COL
].astype(int).to_numpy()

primary_recall = float(
    primary.mean()
)

primary_correct = (
    primary == 1
)


print(
    "\nPrimary recall:",
    primary_recall,
)

print(
    "Primary correct:",
    int(primary.sum()),
    "/",
    len(primary),
)


# ---------------------------------------
# Model-level paired analyses
# ---------------------------------------

rows = []

for model_name, col in (
    ALL_MODELS.items()
):

    pred = (
        q1[col]
        .astype(int)
        .to_numpy()
    )

    recall = float(
        pred.mean()
    )

    delta = (
        recall
        - primary_recall
    )

    ci_lo, ci_hi = (
        paired_bootstrap_delta(
            primary,
            pred,
        )
    )

    model_correct = (
        pred == 1
    )

    b, c, n_disc, p = (
        exact_mcnemar(
            primary_correct,
            model_correct,
        )
    )

    rows.append(
        {
            "model":
                model_name,
            "family":
                (
                    "conventional"
                    if model_name
                    in CONVENTIONAL
                    else "llm"
                ),
            "n":
                len(q1),
            "primary_recall":
                primary_recall,
            "model_recall":
                recall,
            "delta_recall":
                delta,
            "delta_recall_ci_lo":
                ci_lo,
            "delta_recall_ci_hi":
                ci_hi,
            "primary_correct_model_wrong_b":
                b,
            "primary_wrong_model_correct_c":
                c,
            "discordant_pairs":
                n_disc,
            "mcnemar_p_raw":
                p,
        }
    )


result = pd.DataFrame(
    rows
)


# ---------------------------------------
# Holm correction:
# separate pre-existing comparison families
# ---------------------------------------

for family in [
    "conventional",
    "llm",
]:

    mask = (
        result["family"]
        == family
    )

    adjusted = holm_adjust(
        result.loc[
            mask,
            "mcnemar_p_raw",
        ].to_numpy()
    )

    result.loc[
        mask,
        "mcnemar_p_holm",
    ] = adjusted


result[
    "mcnemar_significant_holm_0.05"
] = (
    result[
        "mcnemar_p_holm"
    ]
    < 0.05
)


# ---------------------------------------
# Useful paired interpretation fields
# ---------------------------------------

result[
    "net_correct_gain"
] = (
    result[
        "primary_wrong_model_correct_c"
    ]
    -
    result[
        "primary_correct_model_wrong_b"
    ]
)

result[
    "net_correct_gain_rate"
] = (
    result[
        "net_correct_gain"
    ]
    / result["n"]
)


result.to_csv(
    OUT
    / "q1_paired_recall_statistics.csv",
    index=False,
)


# ---------------------------------------
# Compact table
# ---------------------------------------

compact = result[
    [
        "model",
        "family",
        "model_recall",
        "delta_recall",
        "delta_recall_ci_lo",
        "delta_recall_ci_hi",
        "primary_correct_model_wrong_b",
        "primary_wrong_model_correct_c",
        "net_correct_gain",
        "mcnemar_p_raw",
        "mcnemar_p_holm",
        "mcnemar_significant_holm_0.05",
    ]
].copy()


compact.to_csv(
    OUT
    / "q1_paired_recall_table.csv",
    index=False,
)


# ---------------------------------------
# Similarity summary for frozen Q1
# ---------------------------------------

similarity_summary = {
    "n": int(
        len(q1)
    ),
    "similarity_min": float(
        q1[
            "nearest_train_similarity"
        ].min()
    ),
    "similarity_mean": float(
        q1[
            "nearest_train_similarity"
        ].mean()
    ),
    "similarity_median": float(
        q1[
            "nearest_train_similarity"
        ].median()
    ),
    "similarity_max": float(
        q1[
            "nearest_train_similarity"
        ].max()
    ),
}


with open(
    OUT / "q1_analysis_summary.json",
    "w",
) as f:

    json.dump(
        {
            "analysis_type":
                "secondary explanatory analysis",
            "stratum":
                "external malicious Q1 farthest "
                "by nearest frozen-training "
                "TF-IDF similarity",
            "n_bootstrap":
                N_BOOT,
            "bootstrap_seed":
                SEED,
            "primary_model":
                PRIMARY_NAME,
            "primary_recall":
                primary_recall,
            "similarity_summary":
                similarity_summary,
            "holm_families": {
                "conventional":
                    list(
                        CONVENTIONAL.keys()
                    ),
                "llm":
                    list(
                        LLMS.keys()
                    ),
            },
        },
        f,
        indent=2,
    )


# ---------------------------------------
# LaTeX-ready table
# ---------------------------------------

latex_df = result.copy()

latex_df[
    "Recall"
] = latex_df[
    "model_recall"
].map(
    lambda x: f"{x:.3f}"
)

latex_df[
    "Delta Recall"
] = latex_df.apply(
    lambda r:
    (
        f"{r['delta_recall']:+.3f} "
        f"[{r['delta_recall_ci_lo']:+.3f}, "
        f"{r['delta_recall_ci_hi']:+.3f}]"
    ),
    axis=1,
)

latex_df[
    "Fixes/Harms"
] = latex_df.apply(
    lambda r:
    (
        f"{int(r['primary_wrong_model_correct_c'])}"
        f"/"
        f"{int(r['primary_correct_model_wrong_b'])}"
    ),
    axis=1,
)

latex_df[
    "Holm p"
] = latex_df[
    "mcnemar_p_holm"
].map(
    lambda x:
    (
        f"{x:.3e}"
        if x < 0.001
        else f"{x:.3f}"
    )
)

latex_out = latex_df[
    [
        "model",
        "Recall",
        "Delta Recall",
        "Fixes/Harms",
        "Holm p",
    ]
].rename(
    columns={
        "model": "Model",
    }
)


latex_text = latex_out.to_latex(
    index=False,
    escape=False,
    caption=(
        "Paired performance on the "
        "200 lexically most distant "
        "external malicious scripts."
    ),
    label=(
        "tab:q1_distance_stats"
    ),
)

with open(
    OUT
    / "table_q1_distance_stats.tex",
    "w",
) as f:
    f.write(
        latex_text
    )


# ---------------------------------------
# Console
# ---------------------------------------

print(
    "\n=== Q1 SIMILARITY ==="
)

for k, v in (
    similarity_summary.items()
):
    print(
        k,
        "=",
        v,
    )


print(
    "\n=== PAIRED Q1 RECALL RESULTS ==="
)

print(
    compact.to_string(
        index=False
    )
)


print(
    "\nOutputs:"
)

for p in sorted(
    OUT.iterdir()
):
    print(
        " ",
        p,
    )
