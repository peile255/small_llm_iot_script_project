#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from statsmodels.stats.contingency_tables import mcnemar

OUTDIR = Path("outputs/final")
RNG = np.random.default_rng(42)
N_BOOT = 5000

MODEL_MAP = {
    "mistral_7b": "Mistral-7B-Instruct-v0.3",
    "phi4": "Phi-4",
    "gemma3_27b": "Gemma-3-27B",
    "qwq_32b": "QwQ-32B",
    "deepseek_r1_32b": "DeepSeek-R1-32B",
}


def metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def fast_f1_mcc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    tp = np.sum((y == 1) & (p == 1))
    tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))

    denom_f1 = 2 * tp + fp + fn
    f1 = 0.0 if denom_f1 == 0 else (2 * tp) / denom_f1

    denom_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denom_mcc == 0 else ((tp * tn - fp * fn) / denom_mcc)
    return float(f1), float(mcc)


def bootstrap_ci(y, p, n_boot=N_BOOT):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    n = len(y)

    f1_vals = np.empty(n_boot)
    mcc_vals = np.empty(n_boot)

    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        f1_vals[i], mcc_vals[i] = fast_f1_mcc(y[idx], p[idx])

    f1, mcc = fast_f1_mcc(y, p)
    return {
        "f1": f1,
        "f1_low": np.percentile(f1_vals, 2.5),
        "f1_high": np.percentile(f1_vals, 97.5),
        "mcc": mcc,
        "mcc_low": np.percentile(mcc_vals, 2.5),
        "mcc_high": np.percentile(mcc_vals, 97.5),
    }


def paired_bootstrap_delta_f1(y, baseline_pred, model_pred, n_boot=10000):
    y = np.asarray(y).astype(int)
    b = np.asarray(baseline_pred).astype(int)
    m = np.asarray(model_pred).astype(int)
    n = len(y)

    obs = fast_f1_mcc(y, m)[0] - fast_f1_mcc(y, b)[0]
    deltas = np.empty(n_boot)

    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        deltas[i] = fast_f1_mcc(y[idx], m[idx])[0] - fast_f1_mcc(y[idx], b[idx])[0]

    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(obs), float(low), float(high)


def exact_mcnemar(y, baseline_pred, model_pred):
    y = np.asarray(y).astype(int)
    b = np.asarray(baseline_pred).astype(int)
    m = np.asarray(model_pred).astype(int)

    b_ok = b == y
    m_ok = m == y

    both_correct = int(np.sum(b_ok & m_ok))
    baseline_only = int(np.sum(b_ok & ~m_ok))
    model_only = int(np.sum(~b_ok & m_ok))
    both_wrong = int(np.sum(~b_ok & ~m_ok))

    table = [[both_correct, baseline_only], [model_only, both_wrong]]
    result = mcnemar(table, exact=True)

    return {
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "model_only_correct": model_only,
        "both_wrong": both_wrong,
        "p_value": float(result.pvalue),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(
        OUTDIR / "phi4/test_predictions.csv",
        usecols=["sha256", "truncated", "y_true"],
    )

    baseline = pd.read_csv(OUTDIR / "baseline/predictions.csv")
    baseline_meta = baseline.merge(
        meta[["sha256", "truncated"]],
        on="sha256",
        how="inner",
    )

    # 1) Main model results
    rows = []
    m = metrics(baseline["y_true"], baseline["y_pred"])
    rows.append({
        "model": "Baseline",
        **m,
        "invalid_output_rate": 0.0,
        "llm": False,
    })

    for folder, display in MODEL_MAP.items():
        df = pd.read_csv(OUTDIR / folder / "test_predictions.csv")
        m = metrics(df["y_true"], df["y_pred"])
        rows.append({
            "model": display,
            **m,
            "invalid_output_rate": float((~df["valid"].astype(bool)).mean()),
            "llm": True,
        })

    main_results = pd.DataFrame(rows)
    main_results.to_csv(OUTDIR / "main_model_results.csv", index=False)

    # 2) Length-stratified analysis
    rows = []
    datasets = [("Baseline", baseline_meta)]
    for folder, display in MODEL_MAP.items():
        datasets.append((display, pd.read_csv(OUTDIR / folder / "test_predictions.csv")))

    for model, df in datasets:
        for flag, subset_name in [(False, "<=6000 chars"), (True, ">6000 chars")]:
            sub = df[df["truncated"] == flag]
            m = metrics(sub["y_true"], sub["y_pred"])
            rows.append({
                "model": model,
                "subset": subset_name,
                "positive_n": int(sub["y_true"].astype(int).sum()),
                **m,
            })

    long_results = pd.DataFrame(rows)
    long_results.to_csv(OUTDIR / "long_script_analysis.csv", index=False)

    # 3) Hybrid analysis
    base_h = baseline.merge(meta, on="sha256", how="inner", suffixes=("_base", ""))
    base_h = base_h.rename(columns={"y_pred": "baseline_pred"})

    rows = []
    bm = metrics(base_h["y_true"], base_h["baseline_pred"])
    rows.append({
        "model": "Baseline",
        **bm,
        "llm_calls": 0,
        "llm_call_rate": 0.0,
    })

    for folder in ["phi4", "deepseek_r1_32b", "qwq_32b"]:
        llm = pd.read_csv(
            OUTDIR / folder / "test_predictions.csv",
            usecols=["sha256", "y_pred"],
        ).rename(columns={"y_pred": "llm_pred"})

        df = base_h.merge(llm, on="sha256", how="inner")
        review = (df["baseline_pred"] == 0) & (df["truncated"] == True)

        hybrid_pred = df["baseline_pred"].copy()
        hybrid_pred.loc[review] = df.loc[review, "llm_pred"]

        hm = metrics(df["y_true"], hybrid_pred)
        rows.append({
            "model": f"Hybrid-{MODEL_MAP[folder]}",
            **hm,
            "llm_calls": int(review.sum()),
            "llm_call_rate": float(review.mean()),
        })

    hybrid_results = pd.DataFrame(rows)
    hybrid_results.to_csv(OUTDIR / "hybrid_results.csv", index=False)

    # 4) Bootstrap CI for F1 and MCC
    rows = []
    datasets = [("baseline", baseline_meta)]
    for folder in MODEL_MAP:
        datasets.append((folder, pd.read_csv(OUTDIR / folder / "test_predictions.csv")))

    for model, df in datasets:
        for flag, subset in [(None, "overall"), (False, "<=6000"), (True, ">6000")]:
            sub = df if flag is None else df[df["truncated"] == flag]
            y = sub["y_true"].astype(int).to_numpy()
            p = sub["y_pred"].astype(int).to_numpy()
            ci = bootstrap_ci(y, p)
            rows.append({
                "model": model,
                "subset": subset,
                "n": len(sub),
                **ci,
            })

    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(OUTDIR / "bootstrap_ci_summary.csv", index=False)

    # 5) Paired delta-F1 + McNemar vs baseline
    base_for_pairs = baseline.rename(columns={"y_true": "true", "y_pred": "baseline_pred"}).merge(
        meta[["sha256", "truncated"]],
        on="sha256",
        how="inner",
    )

    paired_rows = []
    mcnemar_rows = []

    for folder in MODEL_MAP:
        pred = pd.read_csv(
            OUTDIR / folder / "test_predictions.csv",
            usecols=["sha256", "y_pred"],
        ).rename(columns={"y_pred": "model_pred"})

        df = base_for_pairs.merge(pred, on="sha256", how="inner")

        for subset_name, sub in [
            ("overall", df),
            (">6000", df[df["truncated"] == True]),
        ]:
            obs, low, high = paired_bootstrap_delta_f1(
                sub["true"],
                sub["baseline_pred"],
                sub["model_pred"],
            )
            paired_rows.append({
                "model": folder,
                "subset": subset_name,
                "delta_f1": obs,
                "ci_low": low,
                "ci_high": high,
            })

            mc = exact_mcnemar(
                sub["true"],
                sub["baseline_pred"],
                sub["model_pred"],
            )
            mcnemar_rows.append({
                "model": folder,
                "subset": subset_name,
                **mc,
            })

    pd.DataFrame(paired_rows).to_csv(
        OUTDIR / "paired_bootstrap_f1.csv",
        index=False,
    )
    pd.DataFrame(mcnemar_rows).to_csv(
        OUTDIR / "mcnemar_results.csv",
        index=False,
    )

    # 6) Error complementarity on long positives
    phi = pd.read_csv(
        OUTDIR / "phi4/test_predictions.csv",
        usecols=["sha256", "y_true", "y_pred", "truncated"],
    ).rename(columns={"y_pred": "phi4_pred"})

    deepseek = pd.read_csv(
        OUTDIR / "deepseek_r1_32b/test_predictions.csv",
        usecols=["sha256", "y_pred"],
    ).rename(columns={"y_pred": "deepseek_pred"})

    err = (
        phi
        .merge(baseline[["sha256", "y_pred"]], on="sha256")
        .rename(columns={"y_pred": "baseline_pred"})
        .merge(deepseek, on="sha256")
    )

    long_pos = err[(err["truncated"] == True) & (err["y_true"] == 1)]
    baseline_missed = long_pos[long_pos["baseline_pred"] == 0]

    error_summary = pd.DataFrame([{
        "long_positive_n": len(long_pos),
        "baseline_missed": len(baseline_missed),
        "recovered_by_phi4": int((baseline_missed["phi4_pred"] == 1).sum()),
        "recovered_by_deepseek": int((baseline_missed["deepseek_pred"] == 1).sum()),
        "recovered_by_both": int(
            ((baseline_missed["phi4_pred"] == 1) &
             (baseline_missed["deepseek_pred"] == 1)).sum()
        ),
    }])
    error_summary.to_csv(OUTDIR / "error_complementarity.csv", index=False)

    print("Saved final analysis files:")
    for name in [
        "main_model_results.csv",
        "long_script_analysis.csv",
        "hybrid_results.csv",
        "bootstrap_ci_summary.csv",
        "paired_bootstrap_f1.csv",
        "mcnemar_results.csv",
        "error_complementarity.csv",
    ]:
        print(" ", OUTDIR / name)


if __name__ == "__main__":
    main()
