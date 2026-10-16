#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


SEED = 20260921
N_BOOT = 10000

ROOT = Path("outputs/paper_v2")
STAT_DIR = ROOT / "statistics"
STAT_DIR.mkdir(parents=True, exist_ok=True)

TEST_MANIFEST = Path(
    "data/processed/final_frozen/final_test.csv"
)

PRIMARY_BASELINE = Path(
    "outputs/final/baseline/predictions.csv"
)

MODEL_FILES = {
    "lr_char": Path(
        "outputs/paper_v2/baselines/lr_char/test_predictions.csv"
    ),
    "svm_char": Path(
        "outputs/paper_v2/baselines/svm_char/test_predictions.csv"
    ),
    "svm_word": Path(
        "outputs/paper_v2/baselines/svm_word/test_predictions.csv"
    ),
    "mistral_7b": Path(
        "outputs/final/mistral_7b/test_predictions.csv"
    ),
    "phi4": Path(
        "outputs/final/phi4/test_predictions.csv"
    ),
    "gemma3_27b": Path(
        "outputs/final/gemma3_27b/test_predictions.csv"
    ),
    "qwq_32b": Path(
        "outputs/final/qwq_32b/test_predictions.csv"
    ),
    "deepseek_r1_32b": Path(
        "outputs/final/deepseek_r1_32b/test_predictions.csv"
    ),
}


def metric_values(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    return (
        float(f1_score(y, pred, zero_division=0)),
        float(matthews_corrcoef(y, pred)),
    )


def build_group_index(groups):
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    members = {
        g: np.flatnonzero(groups == g)
        for g in unique_groups
    }

    return unique_groups, members


def sampled_cluster_indices(
    unique_groups,
    members,
    rng,
):
    sampled_groups = rng.choice(
        unique_groups,
        size=len(unique_groups),
        replace=True,
    )

    return np.concatenate([
        members[g]
        for g in sampled_groups
    ])


def cluster_bootstrap_ci(
    y,
    pred,
    groups,
    n_boot=N_BOOT,
    seed=SEED,
):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    groups = np.asarray(groups)

    unique_groups, members = build_group_index(groups)
    rng = np.random.default_rng(seed)

    f1_boot = np.empty(n_boot)
    mcc_boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = sampled_cluster_indices(
            unique_groups,
            members,
            rng,
        )

        f1_boot[i], mcc_boot[i] = metric_values(
            y[idx],
            pred[idx],
        )

    f1_obs, mcc_obs = metric_values(y, pred)

    return {
        "n": len(y),
        "n_groups": len(unique_groups),
        "f1": f1_obs,
        "f1_low": float(np.percentile(f1_boot, 2.5)),
        "f1_high": float(np.percentile(f1_boot, 97.5)),
        "mcc": mcc_obs,
        "mcc_low": float(np.percentile(mcc_boot, 2.5)),
        "mcc_high": float(np.percentile(mcc_boot, 97.5)),
    }


def paired_cluster_bootstrap_delta_f1(
    y,
    baseline_pred,
    model_pred,
    groups,
    n_boot=N_BOOT,
    seed=SEED,
):
    y = np.asarray(y, dtype=int)
    b = np.asarray(baseline_pred, dtype=int)
    m = np.asarray(model_pred, dtype=int)
    groups = np.asarray(groups)

    unique_groups, members = build_group_index(groups)
    rng = np.random.default_rng(seed)

    delta_boot = np.empty(n_boot)

    obs = (
        f1_score(y, m, zero_division=0)
        -
        f1_score(y, b, zero_division=0)
    )

    for i in range(n_boot):
        idx = sampled_cluster_indices(
            unique_groups,
            members,
            rng,
        )

        delta_boot[i] = (
            f1_score(
                y[idx],
                m[idx],
                zero_division=0,
            )
            -
            f1_score(
                y[idx],
                b[idx],
                zero_division=0,
            )
        )

    return {
        "delta_f1": float(obs),
        "ci_low": float(
            np.percentile(delta_boot, 2.5)
        ),
        "ci_high": float(
            np.percentile(delta_boot, 97.5)
        ),
    }


def exact_mcnemar(
    y,
    baseline_pred,
    model_pred,
):
    y = np.asarray(y, dtype=int)
    b = np.asarray(baseline_pred, dtype=int)
    m = np.asarray(model_pred, dtype=int)

    b_ok = b == y
    m_ok = m == y

    both_correct = int(np.sum(b_ok & m_ok))
    baseline_only = int(np.sum(b_ok & ~m_ok))
    model_only = int(np.sum(~b_ok & m_ok))
    both_wrong = int(np.sum(~b_ok & ~m_ok))

    table = [
        [both_correct, baseline_only],
        [model_only, both_wrong],
    ]

    result = mcnemar(
        table,
        exact=True,
    )

    return {
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "model_only_correct": model_only,
        "both_wrong": both_wrong,
        "p_value": float(result.pvalue),
    }


def load_model_prediction(path):
    df = pd.read_csv(path)

    cols = ["sha256", "y_pred"]

    if "y_true" in df.columns:
        cols.append("y_true")

    return df[cols].copy()


def main():

    manifest = pd.read_csv(
        TEST_MANIFEST,
        usecols=[
            "sha256",
            "label",
            "group_id",
        ],
    ).rename(
        columns={"label": "y_true"}
    )

    baseline = pd.read_csv(
        PRIMARY_BASELINE,
        usecols=[
            "sha256",
            "y_pred",
        ],
    ).rename(
        columns={"y_pred": "baseline_pred"}
    )

    master = manifest.merge(
        baseline,
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    assert len(master) == len(manifest)

    # --------------------------------------------------
    # Cluster-bootstrap CI for every classifier
    # --------------------------------------------------

    ci_rows = []

    primary_ci = cluster_bootstrap_ci(
        master["y_true"],
        master["baseline_pred"],
        master["group_id"],
    )

    ci_rows.append({
        "model": "lr_char_word_primary",
        **primary_ci,
    })

    for model_name, path in MODEL_FILES.items():

        pred = load_model_prediction(path).rename(
            columns={"y_pred": "model_pred"}
        )

        df = master[
            ["sha256", "y_true", "group_id"]
        ].merge(
            pred[["sha256", "model_pred"]],
            on="sha256",
            how="inner",
            validate="one_to_one",
        )

        assert len(df) == len(master)

        ci = cluster_bootstrap_ci(
            df["y_true"],
            df["model_pred"],
            df["group_id"],
        )

        ci_rows.append({
            "model": model_name,
            **ci,
        })

    pd.DataFrame(ci_rows).to_csv(
        STAT_DIR /
        "group_bootstrap_ci.csv",
        index=False,
    )

    # --------------------------------------------------
    # Paired cluster bootstrap vs frozen primary baseline
    # --------------------------------------------------

    paired_rows = []
    mcnemar_rows = []

    for model_name, path in MODEL_FILES.items():

        pred = load_model_prediction(path).rename(
            columns={"y_pred": "model_pred"}
        )

        df = master.merge(
            pred[["sha256", "model_pred"]],
            on="sha256",
            how="inner",
            validate="one_to_one",
        )

        assert len(df) == len(master)

        pair = paired_cluster_bootstrap_delta_f1(
            df["y_true"],
            df["baseline_pred"],
            df["model_pred"],
            df["group_id"],
        )

        paired_rows.append({
            "model": model_name,
            **pair,
        })

        mc = exact_mcnemar(
            df["y_true"],
            df["baseline_pred"],
            df["model_pred"],
        )

        mcnemar_rows.append({
            "model": model_name,
            **mc,
        })

    paired_df = pd.DataFrame(paired_rows)

    paired_df.to_csv(
        STAT_DIR /
        "group_paired_delta_f1.csv",
        index=False,
    )

    mc_df = pd.DataFrame(mcnemar_rows)

    reject, p_holm, _, _ = multipletests(
        mc_df["p_value"].to_numpy(),
        alpha=0.05,
        method="holm",
    )

    mc_df["p_holm"] = p_holm
    mc_df["reject_holm_0_05"] = reject

    mc_df.to_csv(
        STAT_DIR /
        "mcnemar_holm.csv",
        index=False,
    )

    print("Saved:")
    print(
        STAT_DIR /
        "group_bootstrap_ci.csv"
    )
    print(
        STAT_DIR /
        "group_paired_delta_f1.csv"
    )
    print(
        STAT_DIR /
        "mcnemar_holm.csv"
    )


if __name__ == "__main__":
    main()
