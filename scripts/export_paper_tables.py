#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


def main():
    out = Path("outputs/final/paper")
    out.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv("outputs/final/main_model_results.csv")
    long_df = pd.read_csv("outputs/final/long_script_analysis.csv")
    hybrid_df = pd.read_csv("outputs/final/hybrid_results.csv")

    # Table 1
    t1 = main_df[[
        "model", "accuracy", "precision", "recall",
        "f1", "mcc", "invalid_output_rate"
    ]].copy()
    for c in ["accuracy", "precision", "recall", "f1", "mcc", "invalid_output_rate"]:
        t1[c] = t1[c].map(lambda x: f"{x:.4f}")

    tex1 = t1.to_latex(
        index=False,
        escape=False,
        column_format="lcccccc",
        caption="Overall performance of the conventional baseline and local LLMs on the final test set.",
        label="tab:overall_results",
    )
    (out / "table1_overall.tex").write_text(tex1, encoding="utf-8")

    # Table 2
    t2 = long_df[[
        "model", "subset", "n", "positive_n",
        "precision", "recall", "f1", "mcc"
    ]].copy()
    for c in ["precision", "recall", "f1", "mcc"]:
        t2[c] = t2[c].map(lambda x: f"{x:.4f}")

    tex2 = t2.to_latex(
        index=False,
        escape=False,
        column_format="llrrrrrr",
        caption=(
            "Performance stratified by script length. Scripts longer than "
            "6000 characters were represented using the fixed-budget head--tail "
            "strategy for LLM inference."
        ),
        label="tab:length_results",
    )
    (out / "table2_length.tex").write_text(tex2, encoding="utf-8")

    # Table 3
    t3 = hybrid_df[[
        "model", "accuracy", "precision", "recall",
        "f1", "mcc", "fp", "fn", "llm_call_rate"
    ]].copy()
    for c in ["accuracy", "precision", "recall", "f1", "mcc", "llm_call_rate"]:
        t3[c] = t3[c].map(lambda x: f"{x:.4f}")

    tex3 = t3.to_latex(
        index=False,
        escape=False,
        column_format="lcccccrrr",
        caption=(
            "Performance of selective hybrid strategies. The LLM was invoked "
            "only for baseline-negative scripts exceeding 6000 characters."
        ),
        label="tab:hybrid_results",
    )
    (out / "table3_hybrid.tex").write_text(tex3, encoding="utf-8")

    print("Saved LaTeX tables:")
    for f in sorted(out.glob("*.tex")):
        print(" ", f)


if __name__ == "__main__":
    main()
