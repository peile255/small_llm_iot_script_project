#!/usr/bin/env python3

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

SEED = 20260921
N_BOOT = 10000

SUBSET = Path(
    "outputs/paper_v3/"
    "frozen_external_llm_subset_v1/"
    "external_llm_subset_2000.csv"
)

BASE_ROOT = Path(
    "outputs/paper_v3/"
    "external_baselines_v3_1"
)

LLM_ROOT = Path(
    "outputs/paper_v3/"
    "external_llm"
)

OUT = Path(
    "outputs/paper_v3/"
    "external_statistics_v3"
)

OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = "lr_char_word"

CONVENTIONAL = [
    "lr_char_word",
    "lr_char",
    "svm_char",
    "svm_word",
]

LLMS = [
    "mistral_7b",
    "phi4",
    "gemma3_27b",
    "qwq_32b",
    "deepseek_r1_32b",
]

ALL_MODELS = CONVENTIONAL + LLMS


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        p,
        labels=[0, 1],
    ).ravel()

    return {
        "n": len(y),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(
            y, p, zero_division=0
        ),
        "recall": recall_score(
            y, p, zero_division=0
        ),
        "f1": f1_score(
            y, p, zero_division=0
        ),
        "mcc": matthews_corrcoef(y, p),
        "fpr": (
            fp / (fp + tn)
            if fp + tn else np.nan
        ),
        "fnr": (
            fn / (fn + tp)
            if fn + tp else np.nan
        ),
    }


def percentile_ci(values):
    arr = np.asarray(values, dtype=float)

    return (
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    )


def bootstrap_metric_ci(y, p, rng):
    y = np.asarray(y)
    p = np.asarray(p)

    n = len(y)

    f1s = np.empty(N_BOOT)
    mccs = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yy = y[idx]
        pp = p[idx]

        f1s[b] = f1_score(
            yy,
            pp,
            zero_division=0,
        )

        mccs[b] = matthews_corrcoef(
            yy,
            pp,
        )

    f1_lo, f1_hi = percentile_ci(f1s)
    mcc_lo, mcc_hi = percentile_ci(mccs)

    return {
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
        "mcc_ci_low": mcc_lo,
        "mcc_ci_high": mcc_hi,
    }


def paired_bootstrap_delta_f1(
    y,
    p_ref,
    p_alt,
    rng,
):
    y = np.asarray(y)
    p_ref = np.asarray(p_ref)
    p_alt = np.asarray(p_alt)

    n = len(y)

    deltas = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yy = y[idx]

        ref = f1_score(
            yy,
            p_ref[idx],
            zero_division=0,
        )

        alt = f1_score(
            yy,
            p_alt[idx],
            zero_division=0,
        )

        deltas[b] = alt - ref

    lo, hi = percentile_ci(deltas)

    point = (
        f1_score(
            y,
            p_alt,
            zero_division=0,
        )
        -
        f1_score(
            y,
            p_ref,
            zero_division=0,
        )
    )

    return {
        "delta_f1": point,
        "ci_low": lo,
        "ci_high": hi,
    }


def exact_mcnemar(y, p_ref, p_alt):
    y = np.asarray(y)
    p_ref = np.asarray(p_ref)
    p_alt = np.asarray(p_alt)

    ref_ok = p_ref == y
    alt_ok = p_alt == y

    # ref correct, alt wrong
    b = int(
        np.sum(
            ref_ok & (~alt_ok)
        )
    )

    # ref wrong, alt correct
    c = int(
        np.sum(
            (~ref_ok) & alt_ok
        )
    )

    discordant = b + c

    if discordant == 0:
        pvalue = 1.0
    else:
        pvalue = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "ref_correct_alt_wrong": b,
        "ref_wrong_alt_correct": c,
        "discordant": discordant,
        "p_raw": float(pvalue),
    }


def holm_adjust(df):
    df = df.copy()

    m = len(df)

    order = np.argsort(
        df["p_raw"].values
    )

    adjusted = np.empty(m)

    running_max = 0.0

    for rank, pos in enumerate(order):
        factor = m - rank

        value = min(
            1.0,
            factor *
            df.iloc[pos]["p_raw"],
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[pos] = running_max

    df["p_holm"] = adjusted
    df["significant_0_05"] = (
        df["p_holm"] < 0.05
    )

    return df


print("=== LOAD FROZEN SUBSET ===")

subset = pd.read_csv(SUBSET)

assert len(subset) == 2000
assert subset["external_llm_index"].nunique() == 2000
assert subset["external_final_index"].nunique() == 2000
assert subset["label"].value_counts().to_dict() == {
    0: 1000,
    1: 1000,
}

paired = subset[
    [
        "external_llm_index",
        "external_final_index",
        "label",
        "path",
        "normalized_sha256",
        "n_chars",
    ]
].copy()

paired = paired.rename(
    columns={
        "label": "y_true",
        "n_chars": "manifest_chars",
    }
)

print("Subset OK:", len(paired))


print("\n=== MERGE CONVENTIONAL ===")

for model in CONVENTIONAL:
    path = (
        BASE_ROOT /
        model /
        "external_predictions.csv"
    )

    df = pd.read_csv(path)

    assert len(df) == 19632
    assert df["external_final_index"].nunique() == 19632

    tmp = df[
        [
            "external_final_index",
            "y_true",
            "y_pred",
        ]
    ].rename(
        columns={
            "y_true": f"{model}_y_true",
            "y_pred": f"pred_{model}",
        }
    )

    paired = paired.merge(
        tmp,
        on="external_final_index",
        how="left",
        validate="one_to_one",
    )

    assert paired[
        f"pred_{model}"
    ].notna().all()

    assert np.array_equal(
        paired["y_true"].astype(int),
        paired[
            f"{model}_y_true"
        ].astype(int),
    )

    paired.drop(
        columns=[
            f"{model}_y_true"
        ],
        inplace=True,
    )

    paired[
        f"pred_{model}"
    ] = paired[
        f"pred_{model}"
    ].astype(int)

    print(
        model,
        "matched:",
        len(paired),
    )


print("\n=== MERGE LLMs ===")

truncation_vectors = []
original_char_vectors = []

for model in LLMS:
    path = (
        LLM_ROOT /
        model /
        "test_predictions.csv"
    )

    df = pd.read_csv(path)

    assert len(df) == 2000
    assert df["row_index"].nunique() == 2000

    tmp = df[
        [
            "row_index",
            "path",
            "y_true",
            "y_pred",
            "truncated",
            "original_chars",
        ]
    ].copy()

    tmp = tmp.rename(
        columns={
            "row_index":
                "external_llm_index",
            "path":
                f"{model}_path",
            "y_true":
                f"{model}_y_true",
            "y_pred":
                f"pred_{model}",
            "truncated":
                f"{model}_truncated",
            "original_chars":
                f"{model}_original_chars",
        }
    )

    paired = paired.merge(
        tmp,
        on="external_llm_index",
        how="left",
        validate="one_to_one",
    )

    assert paired[
        f"pred_{model}"
    ].notna().all()

    assert np.array_equal(
        paired["y_true"].astype(int),
        paired[
            f"{model}_y_true"
        ].astype(int),
    )

    assert np.array_equal(
        paired["path"].astype(str),
        paired[
            f"{model}_path"
        ].astype(str),
    )

    paired[
        f"pred_{model}"
    ] = paired[
        f"pred_{model}"
    ].astype(int)

    truncation_vectors.append(
        paired[
            f"{model}_truncated"
        ].astype(bool).to_numpy()
    )

    original_char_vectors.append(
        paired[
            f"{model}_original_chars"
        ].to_numpy()
    )

    print(
        model,
        "matched:",
        len(paired),
    )


print("\n=== CHECK COMMON LLM LENGTH STRATA ===")

reference_trunc = truncation_vectors[0]

for vec in truncation_vectors[1:]:
    assert np.array_equal(
        reference_trunc,
        vec,
    )

reference_chars = original_char_vectors[0]

for vec in original_char_vectors[1:]:
    assert np.array_equal(
        reference_chars,
        vec,
    )

paired["llm_truncated"] = reference_trunc
paired["llm_original_chars"] = reference_chars

paired["length_stratum"] = np.where(
    paired["llm_truncated"],
    ">6000",
    "<=6000",
)

print(
    paired[
        "length_stratum"
    ].value_counts()
)

print(
    pd.crosstab(
        paired["length_stratum"],
        paired["y_true"],
    )
)


# Remove redundant merge-check columns.
drop_cols = []

for model in LLMS:
    drop_cols += [
        f"{model}_path",
        f"{model}_y_true",
        f"{model}_truncated",
        f"{model}_original_chars",
    ]

paired.drop(
    columns=drop_cols,
    inplace=True,
)

paired.to_csv(
    OUT / "paired_external_2000_predictions.csv",
    index=False,
)


print("\n=== OVERALL METRICS ===")

overall_rows = []

for model in ALL_MODELS:
    m = metrics(
        paired["y_true"],
        paired[f"pred_{model}"],
    )

    rng = np.random.default_rng(
        SEED
        + ALL_MODELS.index(model)
    )

    ci = bootstrap_metric_ci(
        paired["y_true"].to_numpy(),
        paired[
            f"pred_{model}"
        ].to_numpy(),
        rng,
    )

    row = {
        "model": model,
        **m,
        **ci,
    }

    overall_rows.append(row)

overall = pd.DataFrame(
    overall_rows
)

overall.to_csv(
    OUT / "overall_metrics_with_ci.csv",
    index=False,
)

print(
    overall[
        [
            "model",
            "f1",
            "f1_ci_low",
            "f1_ci_high",
            "mcc",
            "mcc_ci_low",
            "mcc_ci_high",
        ]
    ].to_string(
        index=False
    )
)


print("\n=== PAIRED DELTA F1 VS PRIMARY ===")

delta_rows = []

for i, model in enumerate(
    [
        m
        for m in ALL_MODELS
        if m != PRIMARY
    ]
):
    rng = np.random.default_rng(
        SEED + 100 + i
    )

    d = paired_bootstrap_delta_f1(
        paired["y_true"].to_numpy(),
        paired[
            f"pred_{PRIMARY}"
        ].to_numpy(),
        paired[
            f"pred_{model}"
        ].to_numpy(),
        rng,
    )

    delta_rows.append(
        {
            "reference": PRIMARY,
            "model": model,
            **d,
        }
    )

delta = pd.DataFrame(
    delta_rows
)

delta.to_csv(
    OUT / "paired_delta_f1_vs_primary.csv",
    index=False,
)

print(delta.to_string(index=False))


print("\n=== MCNEMAR OVERALL ===")

def run_mcnemar_family(models, family_name):
    rows = []

    for model in models:
        r = exact_mcnemar(
            paired["y_true"],
            paired[
                f"pred_{PRIMARY}"
            ],
            paired[
                f"pred_{model}"
            ],
        )

        rows.append(
            {
                "family": family_name,
                "reference": PRIMARY,
                "model": model,
                **r,
            }
        )

    df = pd.DataFrame(rows)

    return holm_adjust(df)


mc_conv = run_mcnemar_family(
    [
        "lr_char",
        "svm_char",
        "svm_word",
    ],
    "conventional_robustness",
)

mc_llm = run_mcnemar_family(
    LLMS,
    "llm",
)

mc_all = pd.concat(
    [mc_conv, mc_llm],
    ignore_index=True,
)

mc_all.to_csv(
    OUT / "mcnemar_holm_vs_primary.csv",
    index=False,
)

print(mc_all.to_string(index=False))


print("\n=== LENGTH-STRATIFIED ANALYSIS ===")

stratum_rows = []

for stratum in [
    "<=6000",
    ">6000",
]:
    sub = paired[
        paired["length_stratum"]
        == stratum
    ].copy()

    print(
        "\nSTRATUM",
        stratum,
        "n=",
        len(sub),
        "positives=",
        int(sub["y_true"].sum()),
    )

    for model in ALL_MODELS:
        m = metrics(
            sub["y_true"],
            sub[
                f"pred_{model}"
            ],
        )

        stratum_rows.append(
            {
                "stratum": stratum,
                "model": model,
                **m,
            }
        )

length_metrics = pd.DataFrame(
    stratum_rows
)

length_metrics.to_csv(
    OUT / "length_stratified_metrics.csv",
    index=False,
)


print("\n=== LONG-SCRIPT PAIRED DELTA F1 ===")

long = paired[
    paired["length_stratum"]
    == ">6000"
].copy()

long_delta_rows = []

for i, model in enumerate(
    [
        m
        for m in ALL_MODELS
        if m != PRIMARY
    ]
):
    rng = np.random.default_rng(
        SEED + 500 + i
    )

    d = paired_bootstrap_delta_f1(
        long["y_true"].to_numpy(),
        long[
            f"pred_{PRIMARY}"
        ].to_numpy(),
        long[
            f"pred_{model}"
        ].to_numpy(),
        rng,
    )

    long_delta_rows.append(
        {
            "reference": PRIMARY,
            "model": model,
            "n": len(long),
            "positives":
                int(long["y_true"].sum()),
            **d,
        }
    )

long_delta = pd.DataFrame(
    long_delta_rows
)

long_delta.to_csv(
    OUT /
    "long_paired_delta_f1_vs_primary.csv",
    index=False,
)

print(
    long_delta.to_string(
        index=False
    )
)


print("\n=== LONG-SCRIPT MCNEMAR ===")

def long_mcnemar_family(
    models,
    family_name,
):
    rows = []

    for model in models:
        r = exact_mcnemar(
            long["y_true"],
            long[
                f"pred_{PRIMARY}"
            ],
            long[
                f"pred_{model}"
            ],
        )

        rows.append(
            {
                "family": family_name,
                "reference": PRIMARY,
                "model": model,
                "n": len(long),
                **r,
            }
        )

    return holm_adjust(
        pd.DataFrame(rows)
    )


long_mc = pd.concat(
    [
        long_mcnemar_family(
            [
                "lr_char",
                "svm_char",
                "svm_word",
            ],
            "conventional_robustness",
        ),
        long_mcnemar_family(
            LLMS,
            "llm",
        ),
    ],
    ignore_index=True,
)

long_mc.to_csv(
    OUT /
    "long_mcnemar_holm_vs_primary.csv",
    index=False,
)


print("\n=== SAVE SUMMARY ===")

summary = {
    "seed": SEED,
    "n_bootstrap": N_BOOT,
    "paired_n": int(len(paired)),
    "paired_benign": int(
        (paired["y_true"] == 0).sum()
    ),
    "paired_malicious": int(
        (paired["y_true"] == 1).sum()
    ),
    "short_n": int(
        (
            paired["length_stratum"]
            == "<=6000"
        ).sum()
    ),
    "long_n": int(
        (
            paired["length_stratum"]
            == ">6000"
        ).sum()
    ),
    "long_rate": float(
        (
            paired["length_stratum"]
            == ">6000"
        ).mean()
    ),
    "primary_model": PRIMARY,
    "models": ALL_MODELS,
    "holm_families": {
        "conventional_robustness": [
            "lr_char",
            "svm_char",
            "svm_word",
        ],
        "llm": LLMS,
    },
}

(
    OUT /
    "external_statistics_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print("\n=== ALL EXTERNAL STATISTICS COMPLETE ===")
