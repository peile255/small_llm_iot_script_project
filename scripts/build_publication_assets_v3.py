#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FREEZE = Path(
    "outputs/paper_v3/"
    "frozen_external_results_v1"
)

STAT = FREEZE / "external_statistics_v3"

BASE = FREEZE / "external_baselines_v3_1"

LLM = FREEZE / "external_llm_metrics"

OUT = Path(
    "outputs/paper_v3/publication_v3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DISPLAY = {
    "lr_char_word":
        "LR Char+Word",
    "lr_char":
        "LR Char",
    "svm_char":
        "SVM Char",
    "svm_word":
        "SVM Word",
    "mistral_7b":
        "Mistral-7B",
    "phi4":
        "Phi-4",
    "gemma3_27b":
        "Gemma-3-27B",
    "qwq_32b":
        "QwQ-32B",
    "deepseek_r1_32b":
        "DeepSeek-R1-32B",
}


ORDER = [
    "lr_char_word",
    "lr_char",
    "svm_char",
    "svm_word",
    "mistral_7b",
    "phi4",
    "gemma3_27b",
    "qwq_32b",
    "deepseek_r1_32b",
]


def fmt(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def ci_text(lo, hi):
    return (
        f"[{lo:.3f}, {hi:.3f}]"
    )


def save_tex(
    df,
    path,
    caption,
    label,
    column_format=None,
):
    tex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: (
            f"{x:.3f}"
        ),
        caption=caption,
        label=label,
        column_format=column_format,
    )

    path.write_text(tex)


print("=== VERIFY FREEZE ===")

assert FREEZE.exists()

summary_path = (
    STAT /
    "external_statistics_summary.json"
)

summary = json.loads(
    summary_path.read_text()
)

assert summary["paired_n"] == 2000
assert summary["paired_benign"] == 1000
assert summary["paired_malicious"] == 1000
assert summary["long_n"] == 1417

print(
    "Frozen paired n:",
    summary["paired_n"],
)

print(
    "Long n:",
    summary["long_n"],
)


# ============================================================
# TABLE 1
# Full external conventional evaluation: n=19,632
# ============================================================

print(
    "\n=== TABLE 1: FULL EXTERNAL CONVENTIONAL ==="
)

full = pd.read_csv(
    BASE /
    "external_baseline_summary.csv"
)

full["Model"] = (
    full["model"]
    .map(DISPLAY)
)

table1 = full[
    [
        "Model",
        "precision",
        "recall",
        "f1",
        "mcc",
        "fpr",
        "fnr",
    ]
].copy()

table1.columns = [
    "Model",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "FPR",
    "FNR",
]

table1.to_csv(
    OUT /
    "table_external_full_conventional.csv",
    index=False,
)

save_tex(
    table1,
    OUT /
    "table_external_full_conventional.tex",
    (
        "Performance of conventional classifiers "
        "on the full contamination-controlled "
        "external corpus (n=19,632)."
    ),
    "tab:external_full_conventional",
)

print(
    table1.to_string(
        index=False
    )
)


# ============================================================
# TABLE 2
# Same-sample external comparison: n=2,000
# ============================================================

print(
    "\n=== TABLE 2: PAIRED EXTERNAL 2000 ==="
)

overall = pd.read_csv(
    STAT /
    "overall_metrics_with_ci.csv"
)

overall["order"] = (
    overall["model"]
    .map({
        m: i
        for i, m in enumerate(ORDER)
    })
)

overall = (
    overall
    .sort_values("order")
    .reset_index(drop=True)
)

overall["Model"] = (
    overall["model"]
    .map(DISPLAY)
)

table2 = pd.DataFrame({
    "Model":
        overall["Model"],

    "Precision":
        overall["precision"],

    "Recall":
        overall["recall"],

    "F1":
        overall["f1"],

    "F1 95% CI": [
        ci_text(lo, hi)
        for lo, hi in zip(
            overall["f1_ci_low"],
            overall["f1_ci_high"],
        )
    ],

    "MCC":
        overall["mcc"],

    "MCC 95% CI": [
        ci_text(lo, hi)
        for lo, hi in zip(
            overall["mcc_ci_low"],
            overall["mcc_ci_high"],
        )
    ],

    "FPR":
        overall["fpr"],

    "FNR":
        overall["fnr"],
})

table2.to_csv(
    OUT /
    "table_external_paired_2000.csv",
    index=False,
)

save_tex(
    table2,
    OUT /
    "table_external_paired_2000.tex",
    (
        "Same-sample comparison on the "
        "pre-specified balanced external subset "
        "(n=2,000; 1,000 benign and 1,000 malicious)."
    ),
    "tab:external_paired_2000",
)

print(
    table2.to_string(
        index=False
    )
)


# ============================================================
# TABLE 3
# Paired delta F1 vs primary
# ============================================================

print(
    "\n=== TABLE 3: PAIRED DELTA F1 ==="
)

delta = pd.read_csv(
    STAT /
    "paired_delta_f1_vs_primary.csv"
)

delta["Model"] = (
    delta["model"]
    .map(DISPLAY)
)

delta["CI excludes zero"] = (
    (delta["ci_low"] > 0)
    |
    (delta["ci_high"] < 0)
)

table3 = pd.DataFrame({
    "Model":
        delta["Model"],

    "Delta F1":
        delta["delta_f1"],

    "95% CI": [
        ci_text(lo, hi)
        for lo, hi in zip(
            delta["ci_low"],
            delta["ci_high"],
        )
    ],

    "CI excludes zero":
        delta["CI excludes zero"],
})

table3.to_csv(
    OUT /
    "table_external_delta_f1.csv",
    index=False,
)

save_tex(
    table3,
    OUT /
    "table_external_delta_f1.tex",
    (
        "Paired F1 differences relative to "
        "the primary LR Char+Word comparator "
        "on the balanced external subset."
    ),
    "tab:external_delta_f1",
)

print(
    table3.to_string(
        index=False
    )
)


# ============================================================
# TABLE 4
# External LLM operational metrics
# ============================================================

print(
    "\n=== TABLE 4: EXTERNAL LLM OPERATIONAL ==="
)

llm_rows = []

for model in [
    "mistral_7b",
    "phi4",
    "gemma3_27b",
    "qwq_32b",
    "deepseek_r1_32b",
]:
    obj = json.loads(
        (
            LLM /
            model /
            "metrics.json"
        ).read_text()
    )

    m = obj["test"]

    llm_rows.append({
        "Model":
            DISPLAY[model],

        "Invalid output rate":
            m["invalid_output_rate"],

        "API failure rate":
            m["api_failure_rate"],

        "Truncation rate":
            m["truncation_rate"],

        "Median latency (s)":
            m["median_latency_s"],

        "Throughput (files/s)":
            m["throughput_files_s"],
    })

table4 = pd.DataFrame(
    llm_rows
)

table4.to_csv(
    OUT /
    "table_external_llm_operational.csv",
    index=False,
)

save_tex(
    table4,
    OUT /
    "table_external_llm_operational.tex",
    (
        "Operational characteristics of local "
        "LLMs on the external subset."
    ),
    "tab:external_llm_operational",
)

print(
    table4.to_string(
        index=False
    )
)


# ============================================================
# FIGURE 1
# External same-sample F1 with 95% CI
# ============================================================

print(
    "\n=== FIGURE 1: EXTERNAL F1 CI ==="
)

plot_df = overall.copy()

labels = [
    DISPLAY[x]
    for x in plot_df["model"]
]

y = np.arange(
    len(plot_df)
)

x = (
    plot_df["f1"]
    .to_numpy()
)

lo = (
    plot_df["f1_ci_low"]
    .to_numpy()
)

hi = (
    plot_df["f1_ci_high"]
    .to_numpy()
)

xerr = np.vstack([
    x - lo,
    hi - x,
])

fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)

ax.errorbar(
    x,
    y,
    xerr=xerr,
    fmt="o",
    capsize=4,
)

ax.set_yticks(y)
ax.set_yticklabels(labels)

ax.set_xlabel(
    "F1 score (95% bootstrap CI)"
)

ax.set_title(
    "External Same-Sample Comparison (n=2,000)"
)

ax.set_xlim(
    max(0.0, lo.min() - 0.03),
    min(1.0, hi.max() + 0.03),
)

ax.grid(
    axis="x",
    alpha=0.25,
)

ax.invert_yaxis()

fig.tight_layout()

fig.savefig(
    OUT /
    "fig_external_f1_ci.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT /
    "fig_external_f1_ci.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# Long-script delta F1 vs primary
# ============================================================

print(
    "=== FIGURE 2: LONG DELTA F1 ==="
)

long_delta = pd.read_csv(
    STAT /
    "long_paired_delta_f1_vs_primary.csv"
)

long_delta["Model"] = (
    long_delta["model"]
    .map(DISPLAY)
)

labels = (
    long_delta["Model"]
    .tolist()
)

y = np.arange(
    len(long_delta)
)

x = (
    long_delta["delta_f1"]
    .to_numpy()
)

lo = (
    long_delta["ci_low"]
    .to_numpy()
)

hi = (
    long_delta["ci_high"]
    .to_numpy()
)

xerr = np.vstack([
    x - lo,
    hi - x,
])

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

ax.errorbar(
    x,
    y,
    xerr=xerr,
    fmt="o",
    capsize=4,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

ax.set_yticks(y)
ax.set_yticklabels(labels)

ax.set_xlabel(
    "Paired ΔF1 vs LR Char+Word "
    "(95% bootstrap CI)"
)

ax.set_title(
    "External Scripts Exceeding 6,000 Characters "
    "(n=1,417)"
)

pad = 0.02

ax.set_xlim(
    lo.min() - pad,
    hi.max() + pad,
)

ax.grid(
    axis="x",
    alpha=0.25,
)

ax.invert_yaxis()

fig.tight_layout()

fig.savefig(
    OUT /
    "fig_external_long_delta_f1.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT /
    "fig_external_long_delta_f1.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# COPY STATISTICAL SUPPORT TABLES
# ============================================================

for name in [
    "mcnemar_holm_vs_primary.csv",
    "long_mcnemar_holm_vs_primary.csv",
    "length_stratified_metrics.csv",
]:
    src = STAT / name

    dst = OUT / name

    dst.write_bytes(
        src.read_bytes()
    )


print(
    "\n=== PUBLICATION ASSETS COMPLETE ==="
)

print(
    "Output directory:",
    OUT,
)
