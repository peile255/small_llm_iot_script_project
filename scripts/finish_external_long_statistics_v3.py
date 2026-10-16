#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SEED = 20260921
N_BOOT = 10000

ROOT = Path(
    "outputs/paper_v3/"
    "external_statistics_v3"
)

PAIRED = ROOT / (
    "paired_external_2000_predictions.csv"
)

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


def binary_f1(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.int8)

    tp = np.sum((y == 1) & (p == 1))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))

    denom = 2 * tp + fp + fn

    if denom == 0:
        return 0.0

    return float(2 * tp / denom)


def paired_bootstrap_delta_f1(
    y,
    p_ref,
    p_alt,
    seed,
):
    y = np.asarray(y, dtype=np.int8)
    p_ref = np.asarray(p_ref, dtype=np.int8)
    p_alt = np.asarray(p_alt, dtype=np.int8)

    rng = np.random.default_rng(seed)

    n = len(y)

    deltas = np.empty(
        N_BOOT,
        dtype=np.float64,
    )

    for b in range(N_BOOT):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yy = y[idx]

        ref_f1 = binary_f1(
            yy,
            p_ref[idx],
        )

        alt_f1 = binary_f1(
            yy,
            p_alt[idx],
        )

        deltas[b] = (
            alt_f1 - ref_f1
        )

    point = (
        binary_f1(y, p_alt)
        -
        binary_f1(y, p_ref)
    )

    return {
        "delta_f1": point,
        "ci_low": float(
            np.quantile(
                deltas,
                0.025,
            )
        ),
        "ci_high": float(
            np.quantile(
                deltas,
                0.975,
            )
        ),
    }


def exact_mcnemar(
    y,
    p_ref,
    p_alt,
):
    y = np.asarray(y)
    p_ref = np.asarray(p_ref)
    p_alt = np.asarray(p_alt)

    ref_ok = p_ref == y
    alt_ok = p_alt == y

    b = int(
        np.sum(
            ref_ok & (~alt_ok)
        )
    )

    c = int(
        np.sum(
            (~ref_ok) & alt_ok
        )
    )

    discordant = b + c

    if discordant == 0:
        p_raw = 1.0
    else:
        p_raw = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "ref_correct_alt_wrong": b,
        "ref_wrong_alt_correct": c,
        "discordant": discordant,
        "p_raw": float(p_raw),
    }


def holm_adjust(df):
    df = df.copy()

    p = df["p_raw"].to_numpy()
    m = len(p)

    order = np.argsort(p)
    adjusted = np.empty(m)

    running = 0.0

    for rank, pos in enumerate(order):
        value = min(
            1.0,
            (m - rank) * p[pos],
        )

        running = max(
            running,
            value,
        )

        adjusted[pos] = running

    df["p_holm"] = adjusted

    df["significant_0_05"] = (
        df["p_holm"] < 0.05
    )

    return df


print("=== LOAD PAIRED DATA ===")

df = pd.read_csv(PAIRED)

assert len(df) == 2000

print(
    "Overall n:",
    len(df),
)

print(
    "\nLength strata:"
)

print(
    df[
        "length_stratum"
    ].value_counts()
)

print(
    "\nLength x label:"
)

print(
    pd.crosstab(
        df["length_stratum"],
        df["y_true"],
    )
)


long = df[
    df["length_stratum"]
    == ">6000"
].copy()

print(
    "\nLong n:",
    len(long),
)

print(
    "Long positives:",
    int(long["y_true"].sum()),
)

print(
    "Long negatives:",
    int(
        (long["y_true"] == 0).sum()
    ),
)


print(
    "\n=== FAST LONG PAIRED BOOTSTRAP ==="
)

rows = []

comparison_models = [
    m
    for m in ALL_MODELS
    if m != PRIMARY
]

for i, model in enumerate(
    comparison_models
):
    print(
        f"Bootstrap: {model}"
    )

    result = (
        paired_bootstrap_delta_f1(
            long[
                "y_true"
            ].to_numpy(),
            long[
                f"pred_{PRIMARY}"
            ].to_numpy(),
            long[
                f"pred_{model}"
            ].to_numpy(),
            SEED + 500 + i,
        )
    )

    rows.append(
        {
            "reference": PRIMARY,
            "model": model,
            "n": len(long),
            "positives": int(
                long["y_true"].sum()
            ),
            **result,
        }
    )

long_delta = pd.DataFrame(rows)

long_delta.to_csv(
    ROOT /
    "long_paired_delta_f1_vs_primary.csv",
    index=False,
)

print(
    "\n=== LONG DELTA F1 ==="
)

print(
    long_delta.to_string(
        index=False
    )
)


print(
    "\n=== LONG MCNEMAR ==="
)


def run_family(
    models,
    family,
):
    rows = []

    for model in models:
        result = exact_mcnemar(
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
                "family": family,
                "reference": PRIMARY,
                "model": model,
                "n": len(long),
                **result,
            }
        )

    return holm_adjust(
        pd.DataFrame(rows)
    )


long_mc = pd.concat(
    [
        run_family(
            [
                "lr_char",
                "svm_char",
                "svm_word",
            ],
            "conventional_robustness",
        ),
        run_family(
            LLMS,
            "llm",
        ),
    ],
    ignore_index=True,
)

long_mc.to_csv(
    ROOT /
    "long_mcnemar_holm_vs_primary.csv",
    index=False,
)

print(
    long_mc.to_string(
        index=False
    )
)


summary = {
    "seed": SEED,
    "n_bootstrap": N_BOOT,
    "paired_n": int(len(df)),
    "paired_benign": int(
        (df["y_true"] == 0).sum()
    ),
    "paired_malicious": int(
        (df["y_true"] == 1).sum()
    ),
    "short_n": int(
        (
            df["length_stratum"]
            == "<=6000"
        ).sum()
    ),
    "long_n": int(
        (
            df["length_stratum"]
            == ">6000"
        ).sum()
    ),
    "long_rate": float(
        (
            df["length_stratum"]
            == ">6000"
        ).mean()
    ),
    "long_positive": int(
        long["y_true"].sum()
    ),
    "long_negative": int(
        (long["y_true"] == 0).sum()
    ),
    "primary_model": PRIMARY,
    "models": ALL_MODELS,
    "bootstrap_type":
        "paired sample-level percentile bootstrap",
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
    ROOT /
    "external_statistics_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

print(
    "\n=== SUMMARY ==="
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print(
    "\n=== EXTERNAL LONG ANALYSIS COMPLETE ==="
)
