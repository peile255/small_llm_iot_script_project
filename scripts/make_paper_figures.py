#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    out = Path("outputs/final/figures")
    out.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv("outputs/final/main_model_results.csv")
    long_df = pd.read_csv("outputs/final/long_script_analysis.csv")
    hybrid_df = pd.read_csv("outputs/final/hybrid_results.csv")

    # Figure 1: Overall F1
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(main_df["model"], main_df["f1"])
    ax.set_ylabel("F1 score")
    ax.set_title("Overall F1 Performance")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(main_df["f1"]):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_overall_f1.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig1_overall_f1.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Precision-recall
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(main_df["recall"], main_df["precision"], s=90)
    for _, r in main_df.iterrows():
        ax.annotate(
            r["model"],
            (r["recall"], r["precision"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Trade-off")
    ax.set_xlim(0.85, 1.01)
    ax.set_ylim(0.40, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_precision_recall.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig2_precision_recall.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Short vs long F1
    pivot = long_df.pivot(index="model", columns="subset", values="f1")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("")
    ax.set_title("F1 Performance by Script Length")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Script length")
    fig.tight_layout()
    fig.savefig(out / "fig3_length_f1.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig3_length_f1.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: FP-FN trade-off
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(hybrid_df["fp"], hybrid_df["fn"], s=100)
    for _, r in hybrid_df.iterrows():
        ax.annotate(
            r["model"],
            (r["fp"], r["fn"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("False Positives")
    ax.set_ylabel("False Negatives")
    ax.set_title("FP-FN Trade-off of Selective Hybrid Methods")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_hybrid_fp_fn.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig4_hybrid_fp_fn.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved figures:")
    for f in sorted(out.iterdir()):
        print(" ", f)


if __name__ == "__main__":
    main()
