#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd


OUT_ROOT = Path("outputs/paper_v2")
TABLE_DIR = OUT_ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_SUMMARY = Path(
    "outputs/paper_v2/baselines/baseline_v2_summary.csv"
)

LLM_METRICS = {
    "Mistral-7B-Instruct-v0.3":
        Path("outputs/final/mistral_7b/metrics.json"),
    "Phi-4":
        Path("outputs/final/phi4/metrics.json"),
    "Gemma-3-27B":
        Path("outputs/final/gemma3_27b/metrics.json"),
    "QwQ-32B":
        Path("outputs/final/qwq_32b/metrics.json"),
    "DeepSeek-R1-32B":
        Path("outputs/final/deepseek_r1_32b/metrics.json"),
}

BASELINE_NAMES = {
    "lr_char_word": "LR char+word (frozen)",
    "lr_char": "LR char",
    "svm_char": "Linear SVM char",
    "svm_word": "Linear SVM word",
}


def main():
    rows = []

    # ---------------------------------------
    # Conventional models
    # ---------------------------------------
    base = pd.read_csv(BASELINE_SUMMARY)

    for _, r in base.iterrows():
        rows.append({
            "family": "Conventional",
            "model": BASELINE_NAMES[r["model"]],
            "f1": float(r["f1"]),
            "recall": float(r["recall"]),
            "precision": float(r["precision"]),
            "mcc": float(r["mcc"]),
            "latency_s": float(r["mean_latency_s"]),
            "latency_statistic": "mean",
            "throughput_files_s":
                float(r["throughput_files_s"]),
            "invalid_output_rate": 0.0,
        })

    # ---------------------------------------
    # LLMs
    # ---------------------------------------
    for model_name, path in LLM_METRICS.items():
        data = json.loads(path.read_text())
        t = data["test"]

        rows.append({
            "family": "Local LLM",
            "model": model_name,
            "f1": float(t["f1"]),
            "recall": float(t["recall"]),
            "precision": float(t["precision"]),
            "mcc": float(t["mcc"]),
            "latency_s": float(t["median_latency_s"]),
            "latency_statistic": "median",
            "throughput_files_s":
                float(t["throughput_files_s"]),
            "invalid_output_rate":
                float(t["invalid_output_rate"]),
        })

    df = pd.DataFrame(rows)

    csv_path = TABLE_DIR / "operational_results.csv"
    df.to_csv(csv_path, index=False)

    # ---------------------------------------
    # LaTeX table
    # ---------------------------------------
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Predictive and operational characteristics. "
               r"Latency statistics differ by model family: conventional "
               r"classifiers report mean per-file latency over batch "
               r"inference, whereas LLMs report the median of per-sample "
               r"latencies recorded during Ollama inference.}")
    tex.append(r"\label{tab:operational}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{llrrrrrr}")
    tex.append(r"\toprule")
    tex.append(
        r"Family & Model & F1 & Recall & MCC & "
        r"Latency (s) & Files/s & Invalid \\"
    )
    tex.append(r"\midrule")

    for family in ["Conventional", "Local LLM"]:
        sub = df[df["family"] == family]

        for _, r in sub.iterrows():
            invalid = 100.0 * r["invalid_output_rate"]

            tex.append(
                f"{family} & "
                f"{r['model']} & "
                f"{r['f1']:.4f} & "
                f"{r['recall']:.4f} & "
                f"{r['mcc']:.4f} & "
                f"{r['latency_s']:.4f} "
                f"({r['latency_statistic']}) & "
                f"{r['throughput_files_s']:.2f} & "
                f"{invalid:.2f}\\% \\\\"
            )

        if family == "Conventional":
            tex.append(r"\midrule")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")

    tex_path = TABLE_DIR / "table_operational_v2.tex"
    tex_path.write_text(
        "\n".join(tex) + "\n"
    )

    print("Saved:")
    print(csv_path)
    print(tex_path)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
