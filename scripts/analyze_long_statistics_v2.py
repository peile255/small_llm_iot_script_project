#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


SEED = 20260921
N_BOOT = 10000

STAT_DIR = Path("outputs/paper_v2/statistics")
STAT_DIR.mkdir(parents=True, exist_ok=True)

TEST_MANIFEST = Path(
    "data/processed/final_frozen/final_test.csv"
)

BASELINE_FILE = Path(
    "outputs/final/baseline/predictions.csv"
)

PHI_META_FILE = Path(
    "outputs/final/phi4/test_predictions.csv"
)

LLM_FILES = {
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


def fast_f1_mcc(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tp = np.sum((y == 1) & (pred == 1))
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))

    f1_denom = 2 * tp + fp + fn
    f1 = 0.0 if f1_denom == 0 else (2 * tp) / f1_denom

    mcc_denom = np.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        0.0
        if mcc_denom == 0
        else (tp * tn - fp * fn) / mcc_denom
    )

    return float(f1), float(mcc)


def build_group_members(groups):
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    members = {
        g: np.flatnonzero(groups == g)
        for g in unique_groups
    }

    return unique_groups, members


def sample_cluster_indices(
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


def cluster_ci(
    y,
    pred,
    groups,
    n_boot=N_BOOT,
    seed=SEED,
):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    groups = np.asarray(groups)

    unique_groups, members = build_group_members(groups)
    rng = np.random.default_rng(seed)

    f1_boot = np.empty(n_boot)
    mcc_boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = sample_cluster_indices(
            unique_groups,
            members,
            rng,
        )

        f1_boot[i], mcc_boot[i] = fast_f1_mcc(
            y[idx],
            pred[idx],
        )

    f1_obs, mcc_obs = fast_f1_mcc(y, pred)

    return {
        "n": len(y),
        "positive_n": int(np.sum(y == 1)),
        "n_groups": len(unique_groups),
        "f1": f1_obs,
        "f1_low": float(np.percentile(f1_boot, 2.5)),
        "f1_high": float(np.percentile(f1_boot, 97.5)),
        "mcc": mcc_obs,
        "mcc_low": float(np.percentile(mcc_boot, 2.5)),
        "mcc_high": float(np.percentile(mcc_boot, 97.5)),
    }


def paired_cluster_delta_f1(
    y,
    baseline_pred,
    model_pred,
    groups,
    n_boot=N_BOOT,
    seed=SEED,
):
    y = np.asarray(y, dtype=int)
    baseline_pred = np.asarray(
        baseline_pred,
        dtype=int,
    )
    model_pred = np.asarray(
        model_pred,
        dtype=int,
    )
    groups = np.asarray(groups)

    unique_groups, members = build_group_members(groups)
    rng = np.random.default_rng(seed)

    observed = (
        fast_f1_mcc(y, model_pred)[0]
        -
        fast_f1_mcc(y, baseline_pred)[0]
    )

    boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = sample_cluster_indices(
            unique_groups,
            members,
            rng,
        )

        boot[i] = (
            fast_f1_mcc(
                y[idx],
                model_pred[idx],
            )[0]
            -
            fast_f1_mcc(
                y[idx],
                baseline_pred[idx],
            )[0]
        )

    return {
        "delta_f1": float(observed),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
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

    result = mcnemar(
        [
            [both_correct, baseline_only],
            [model_only, both_wrong],
        ],
        exact=True,
    )

    return {
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "model_only_correct": model_only,
        "both_wrong": both_wrong,
        "p_value": float(result.pvalue),
    }


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

    meta = pd.read_csv(
        PHI_META_FILE,
        usecols=[
            "sha256",
            "truncated",
        ],
    )

    baseline = pd.read_csv(
        BASELINE_FILE,
        usecols=[
            "sha256",
            "y_pred",
        ],
    ).rename(
        columns={
            "y_pred": "baseline_pred"
        }
    )

    master = (
        manifest
        .merge(
            meta,
            on="sha256",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            baseline,
            on="sha256",
            how="inner",
            validate="one_to_one",
        )
    )

    long_master = master[
        master["truncated"].astype(bool)
    ].copy()

    assert len(long_master) == 255
    assert int(long_master["y_true"].sum()) == 23

    print(
        "Long subset:",
        len(long_master),
        "samples;",
        int(long_master["y_true"].sum()),
        "positives;",
        long_master["group_id"].nunique(),
        "groups",
    )

    # -----------------------------------------
    # Baseline + LLM group CIs
    # -----------------------------------------

    ci_rows = []

    baseline_ci = cluster_ci(
        long_master["y_true"],
        long_master["baseline_pred"],
        long_master["group_id"],
    )

    ci_rows.append({
        "model": "lr_char_word_primary",
        **baseline_ci,
    })

    paired_rows = []
    mc_rows = []

    for model_name, path in LLM_FILES.items():

        pred = pd.read_csv(
            path,
            usecols=[
                "sha256",
                "y_pred",
            ],
        ).rename(
            columns={
                "y_pred": "model_pred"
            }
        )

        df = long_master.merge(
            pred,
            on="sha256",
            how="inner",
            validate="one_to_one",
        )

        assert len(df) == len(long_master)

        ci = cluster_ci(
            df["y_true"],
            df["model_pred"],
            df["group_id"],
        )

        ci_rows.append({
            "model": model_name,
            **ci,
        })

        pair = paired_cluster_delta_f1(
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

        mc_rows.append({
            "model": model_name,
            **mc,
        })

    ci_df = pd.DataFrame(ci_rows)
    paired_df = pd.DataFrame(paired_rows)
    mc_df = pd.DataFrame(mc_rows)

    reject, p_holm, _, _ = multipletests(
        mc_df["p_value"].values,
        alpha=0.05,
        method="holm",
    )

    mc_df["p_holm"] = p_holm
    mc_df["reject_holm_0_05"] = reject

    ci_path = (
        STAT_DIR /
        "long_group_bootstrap_ci.csv"
    )

    paired_path = (
        STAT_DIR /
        "long_group_paired_delta_f1.csv"
    )

    mc_path = (
        STAT_DIR /
        "long_mcnemar_holm.csv"
    )

    ci_df.to_csv(
        ci_path,
        index=False,
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    mc_df.to_csv(
        mc_path,
        index=False,
    )

    print("\nSaved:")
    print(ci_path)
    print(paired_path)
    print(mc_path)


if __name__ == "__main__":
    main()
