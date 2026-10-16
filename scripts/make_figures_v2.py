#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("outputs/paper_v2/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL = Path(
    "outputs/paper_v2/statistics/group_bootstrap_ci.csv"
)

LONG = Path(
    "outputs/paper_v2/statistics/long_group_bootstrap_ci.csv"
)


DISPLAY = {
    "lr_char_word_primary": "LR char+word (frozen)",
    "lr_char": "LR char",
    "svm_char": "Linear SVM char",
    "svm_word": "Linear SVM word",
    "mistral_7b": "Mistral-7B",
    "phi4": "Phi-4",
    "gemma3_27b": "Gemma-3-27B",
    "qwq_32b": "QwQ-32B",
    "deepseek_r1_32b": "DeepSeek-R1-32B",
}


def figure_overall():
    df = pd.read_csv(OVERALL)

    order = [
        "lr_char_word_primary",
        "lr_char",
        "svm_char",
        "svm_word",
        "mistral_7b",
        "phi4",
        "gemma3_27b",
        "qwq_32b",
        "deepseek_r1_32b",
    ]

    df = (
        df.set_index("model")
        .loc[order]
        .reset_index()
    )

    df["label"] = df["model"].map(DISPLAY)

    lower = df["f1"] - df["f1_low"]
    upper = df["f1_high"] - df["f1"]

    fig, ax = plt.subplots(figsize=(8.5, 6))

    y = range(len(df))

    ax.errorbar(
        df["f1"],
        y,
        xerr=[lower, upper],
        fmt="o",
        capsize=4,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()

    ax.set_xlabel("F1 score")
    ax.set_title(
        "Overall F1 with Group-Bootstrap 95% Confidence Intervals"
    )

    ax.set_xlim(0.50, 1.00)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "fig1_overall_f1_group_ci.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_DIR / "fig1_overall_f1_group_ci.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def figure_long():
    df = pd.read_csv(LONG)

    order = [
        "lr_char_word_primary",
        "mistral_7b",
        "phi4",
        "gemma3_27b",
        "qwq_32b",
        "deepseek_r1_32b",
    ]

    df = (
        df.set_index("model")
        .loc[order]
        .reset_index()
    )

    df["label"] = df["model"].map(DISPLAY)

    lower = df["f1"] - df["f1_low"]
    upper = df["f1_high"] - df["f1"]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    y = range(len(df))

    ax.errorbar(
        df["f1"],
        y,
        xerr=[lower, upper],
        fmt="o",
        capsize=4,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()

    ax.set_xlabel("F1 score")
    ax.set_title(
        "Long-Script F1 with Group-Bootstrap 95% Confidence Intervals"
    )

    ax.set_xlim(0.10, 1.00)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "fig3_long_f1_group_ci.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_DIR / "fig3_long_f1_group_ci.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    figure_overall()
    figure_long()

    print("Saved:")
    for p in sorted(OUT_DIR.glob("*group_ci*")):
        print(p)


if __name__ == "__main__":
    main()
