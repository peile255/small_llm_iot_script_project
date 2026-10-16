from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


SEED = 20260921
N_BOOT = 10000

OUTDIR = Path(
    "outputs/paper_v4/codebert_statistics_v1"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

CODEBERT_DIR = Path(
    "outputs/paper_v4/frozen_codebert_v1"
)

INTERNAL_PRIMARY = Path(
    "outputs/paper_v2/baselines/"
    "lr_char_word/test_predictions.csv"
)

EXTERNAL_MATRIX = Path(
    "outputs/paper_v3/external_statistics_v3/"
    "paired_external_2000_predictions.csv"
)

Q1_FILE = Path(
    "outputs/paper_v4/"
    "frozen_distribution_shift_v1/"
    "q1_statistics/"
    "q1_farthest_malicious_200.csv"
)

Q1_EXISTING_STATS = Path(
    "outputs/paper_v4/"
    "frozen_distribution_shift_v1/"
    "q1_statistics/"
    "q1_paired_recall_table.csv"
)


def find_col(df, candidates, description):
    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        f"Could not find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(y, pred)
        ),
    }


def paired_mcnemar(y, primary, model):
    y = np.asarray(y, dtype=int)
    primary = np.asarray(primary, dtype=int)
    model = np.asarray(model, dtype=int)

    primary_correct = primary == y
    model_correct = model == y

    # b: primary correct, model wrong
    b = int(
        np.sum(
            primary_correct
            & ~model_correct
        )
    )

    # c: primary wrong, model correct
    c = int(
        np.sum(
            ~primary_correct
            & model_correct
        )
    )

    n_disc = b + c

    if n_disc == 0:
        p = 1.0
    else:
        p = float(
            binomtest(
                min(b, c),
                n=n_disc,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "primary_correct_model_wrong_b": b,
        "primary_wrong_model_correct_c": c,
        "net_correct_gain": c - b,
        "discordant_total": n_disc,
        "mcnemar_p_raw": p,
    }


def bootstrap_delta_f1(
    y,
    primary,
    model,
    n_boot=N_BOOT,
    seed=SEED,
):
    y = np.asarray(y, dtype=int)
    primary = np.asarray(primary, dtype=int)
    model = np.asarray(model, dtype=int)

    rng = np.random.default_rng(seed)
    n = len(y)

    deltas = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yp = y[idx]
        pp = primary[idx]
        mp = model[idx]

        deltas[i] = (
            f1_score(
                yp,
                mp,
                zero_division=0,
            )
            - f1_score(
                yp,
                pp,
                zero_division=0,
            )
        )

    point = (
        f1_score(
            y,
            model,
            zero_division=0,
        )
        - f1_score(
            y,
            primary,
            zero_division=0,
        )
    )

    lo, hi = np.percentile(
        deltas,
        [2.5, 97.5],
    )

    return {
        "delta_f1": float(point),
        "delta_f1_ci_lo": float(lo),
        "delta_f1_ci_hi": float(hi),
        "n_bootstrap": int(n_boot),
        "bootstrap_seed": int(seed),
    }


def bootstrap_delta_recall_positive(
    primary,
    model,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    Q1 contains malicious samples only.
    Therefore recall = mean(pred == 1).
    """
    primary = np.asarray(
        primary,
        dtype=int,
    )
    model = np.asarray(
        model,
        dtype=int,
    )

    rng = np.random.default_rng(seed)
    n = len(primary)

    deltas = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        deltas[i] = (
            model[idx].mean()
            - primary[idx].mean()
        )

    point = (
        model.mean()
        - primary.mean()
    )

    lo, hi = np.percentile(
        deltas,
        [2.5, 97.5],
    )

    return {
        "delta_recall": float(point),
        "delta_recall_ci_lo": float(lo),
        "delta_recall_ci_hi": float(hi),
        "n_bootstrap": int(n_boot),
        "bootstrap_seed": int(seed),
    }


def load_internal_primary():
    df = pd.read_csv(
        INTERNAL_PRIMARY
    )

    print(
        "\nInternal primary columns:",
        list(df.columns),
    )

    pred_col = find_col(
        df,
        [
            "y_pred",
            "prediction",
            "pred",
            "pred_lr_char_word",
            "predicted_label",
        ],
        "internal primary prediction column",
    )

    y_col = find_col(
        df,
        [
            "y_true",
            "label",
            "true_label",
        ],
        "internal primary true-label column",
    )

    return df, y_col, pred_col


def align_internal(
    primary_df,
    codebert_df,
):
    """
    Prefer SHA256 alignment.
    Fall back to path alignment.
    Row order is used only after strict
    equality checks.
    """

    if (
        "sha256" in primary_df.columns
        and "sha256" in codebert_df.columns
    ):
        left = primary_df.copy()

        keep = [
            "sha256",
            "y_true_primary",
            "pred_primary",
        ]

        aligned_primary = pd.DataFrame({
            "sha256":
                left["sha256"].astype(str),
            "y_true_primary":
                left["_y_internal"].astype(int),
            "pred_primary":
                left["_pred_internal"].astype(int),
        })

        cb = codebert_df[
            [
                "sha256",
                "y_true",
                "y_pred_codebert",
            ]
        ].copy()

        cb["sha256"] = (
            cb["sha256"].astype(str)
        )

        merged = aligned_primary.merge(
            cb,
            on="sha256",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(codebert_df):
            raise RuntimeError(
                "Internal SHA256 alignment "
                f"returned {len(merged)} rows, "
                f"expected {len(codebert_df)}."
            )

        if not np.array_equal(
            merged[
                "y_true_primary"
            ].to_numpy(),
            merged[
                "y_true"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "Internal labels differ after "
                "SHA256 alignment."
            )

        return merged

    if (
        "path" in primary_df.columns
        and "path" in codebert_df.columns
    ):
        aligned_primary = pd.DataFrame({
            "path":
                primary_df[
                    "path"
                ].astype(str),
            "y_true_primary":
                primary_df[
                    "_y_internal"
                ].astype(int),
            "pred_primary":
                primary_df[
                    "_pred_internal"
                ].astype(int),
        })

        cb = codebert_df[
            [
                "path",
                "y_true",
                "y_pred_codebert",
            ]
        ].copy()

        cb["path"] = cb["path"].astype(str)

        merged = aligned_primary.merge(
            cb,
            on="path",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(codebert_df):
            raise RuntimeError(
                "Internal path alignment "
                f"returned {len(merged)} rows, "
                f"expected {len(codebert_df)}."
            )

        if not np.array_equal(
            merged[
                "y_true_primary"
            ].to_numpy(),
            merged["y_true"].to_numpy(),
        ):
            raise RuntimeError(
                "Internal labels differ after "
                "path alignment."
            )

        return merged

    if len(primary_df) != len(codebert_df):
        raise RuntimeError(
            "No usable identifier for "
            "internal alignment."
        )

    if not np.array_equal(
        primary_df[
            "_y_internal"
        ].astype(int).to_numpy(),
        codebert_df[
            "y_true"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            "Internal row-order labels differ."
        )

    print(
        "WARNING: internal comparison "
        "uses row-order alignment."
    )

    return pd.DataFrame({
        "y_true":
            codebert_df[
                "y_true"
            ].astype(int),
        "pred_primary":
            primary_df[
                "_pred_internal"
            ].astype(int),
        "y_pred_codebert":
            codebert_df[
                "y_pred_codebert"
            ].astype(int),
    })


def analyze_internal():
    print(
        "\n=== INTERNAL PAIRED ANALYSIS ==="
    )

    cb = pd.read_csv(
        CODEBERT_DIR
        / "internal_test_predictions.csv"
    )

    primary, y_col, pred_col = (
        load_internal_primary()
    )

    primary["_y_internal"] = (
        primary[y_col].astype(int)
    )
    primary["_pred_internal"] = (
        primary[pred_col].astype(int)
    )

    merged = align_internal(
        primary,
        cb,
    )

    y = merged["y_true"].astype(int)
    p0 = merged[
        "pred_primary"
    ].astype(int)
    p1 = merged[
        "y_pred_codebert"
    ].astype(int)

    primary_metrics = binary_metrics(
        y,
        p0,
    )

    codebert_metrics = binary_metrics(
        y,
        p1,
    )

    stats = {}
    stats.update(
        bootstrap_delta_f1(
            y,
            p0,
            p1,
        )
    )
    stats.update(
        paired_mcnemar(
            y,
            p0,
            p1,
        )
    )

    row = {
        "comparison":
            "CodeBERT - LR Char+Word",
        "primary_f1":
            primary_metrics["f1"],
        "codebert_f1":
            codebert_metrics["f1"],
        "primary_mcc":
            primary_metrics["mcc"],
        "codebert_mcc":
            codebert_metrics["mcc"],
        **stats,
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        OUTDIR
        / "internal_codebert_paired_stats.csv",
        index=False,
    )

    print(
        json.dumps(
            {
                "primary":
                    primary_metrics,
                "codebert":
                    codebert_metrics,
                "paired":
                    stats,
            },
            indent=2,
        )
    )

    return {
        "primary":
            primary_metrics,
        "codebert":
            codebert_metrics,
        "paired":
            stats,
    }


def analyze_external():
    print(
        "\n=== EXTERNAL 2000 PAIRED ANALYSIS ==="
    )

    matrix = pd.read_csv(
        EXTERNAL_MATRIX
    )

    cb = pd.read_csv(
        CODEBERT_DIR
        / "external_2000_predictions.csv"
    )

    needed = [
        "external_llm_index",
        "y_true",
        "y_pred_codebert",
    ]

    cb_small = cb[
        needed
    ].copy()

    merged = matrix.merge(
        cb_small,
        on="external_llm_index",
        suffixes=("_matrix", "_codebert"),
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 2000:
        raise RuntimeError(
            "External merge returned "
            f"{len(merged)}, expected 2000."
        )

    if not np.array_equal(
        merged[
            "y_true_matrix"
        ].astype(int).to_numpy(),
        merged[
            "y_true_codebert"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            "External y_true mismatch."
        )

    y = merged[
        "y_true_matrix"
    ].astype(int)

    p0 = merged[
        "pred_lr_char_word"
    ].astype(int)

    p1 = merged[
        "y_pred_codebert"
    ].astype(int)

    primary_metrics = binary_metrics(
        y,
        p0,
    )

    codebert_metrics = binary_metrics(
        y,
        p1,
    )

    stats = {}
    stats.update(
        bootstrap_delta_f1(
            y,
            p0,
            p1,
        )
    )
    stats.update(
        paired_mcnemar(
            y,
            p0,
            p1,
        )
    )

    row = {
        "comparison":
            "CodeBERT - LR Char+Word",
        "primary_f1":
            primary_metrics["f1"],
        "codebert_f1":
            codebert_metrics["f1"],
        "primary_mcc":
            primary_metrics["mcc"],
        "codebert_mcc":
            codebert_metrics["mcc"],
        **stats,
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        OUTDIR
        / "external_codebert_paired_stats.csv",
        index=False,
    )

    merged.to_csv(
        OUTDIR
        / "paired_external_2000_with_codebert.csv",
        index=False,
    )

    print(
        json.dumps(
            {
                "primary":
                    primary_metrics,
                "codebert":
                    codebert_metrics,
                "paired":
                    stats,
            },
            indent=2,
        )
    )

    return {
        "primary":
            primary_metrics,
        "codebert":
            codebert_metrics,
        "paired":
            stats,
    }


def analyze_q1():
    print(
        "\n=== Q1 FARTHEST MALICIOUS ANALYSIS ==="
    )

    q1 = pd.read_csv(
        Q1_FILE
    )

    cb = pd.read_csv(
        CODEBERT_DIR
        / "external_2000_predictions.csv"
    )

    cb_small = cb[
        [
            "external_llm_index",
            "y_true",
            "y_pred_codebert",
        ]
    ].copy()

    merged = q1.merge(
        cb_small,
        on="external_llm_index",
        suffixes=("_q1", "_codebert"),
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 200:
        raise RuntimeError(
            "Q1 merge returned "
            f"{len(merged)}, expected 200."
        )

    if not np.all(
        merged[
            "y_true_q1"
        ].astype(int).to_numpy()
        == 1
    ):
        raise RuntimeError(
            "Q1 contains non-malicious labels."
        )

    if not np.array_equal(
        merged[
            "y_true_q1"
        ].astype(int).to_numpy(),
        merged[
            "y_true_codebert"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            "Q1 y_true mismatch."
        )

    primary = merged[
        "pred_lr_char_word"
    ].astype(int).to_numpy()

    codebert = merged[
        "y_pred_codebert"
    ].astype(int).to_numpy()

    primary_recall = float(
        primary.mean()
    )

    codebert_recall = float(
        codebert.mean()
    )

    boot = (
        bootstrap_delta_recall_positive(
            primary,
            codebert,
        )
    )

    y = np.ones(
        len(primary),
        dtype=int,
    )

    mc = paired_mcnemar(
        y,
        primary,
        codebert,
    )

    row = {
        "model":
            "CodeBERT",
        "family":
            "pretrained_discriminative",
        "model_recall":
            codebert_recall,
        **boot,
        **mc,
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        OUTDIR
        / "q1_codebert_paired_stats.csv",
        index=False,
    )

    merged.to_csv(
        OUTDIR
        / "q1_farthest_malicious_with_codebert.csv",
        index=False,
    )

    # Append CodeBERT to existing Q1 table
    # without changing the previously frozen
    # Holm-adjusted p-values of the two
    # pre-specified families.
    existing = pd.read_csv(
        Q1_EXISTING_STATS
    )

    cb_row = {
        "model":
            "CodeBERT",
        "family":
            "pretrained_discriminative",
        "model_recall":
            codebert_recall,
        "delta_recall":
            boot["delta_recall"],
        "delta_recall_ci_lo":
            boot[
                "delta_recall_ci_lo"
            ],
        "delta_recall_ci_hi":
            boot[
                "delta_recall_ci_hi"
            ],
        "primary_correct_model_wrong_b":
            mc[
                "primary_correct_model_wrong_b"
            ],
        "primary_wrong_model_correct_c":
            mc[
                "primary_wrong_model_correct_c"
            ],
        "net_correct_gain":
            mc[
                "net_correct_gain"
            ],
        "mcnemar_p_raw":
            mc[
                "mcnemar_p_raw"
            ],
        # Deliberately NA:
        # CodeBERT is a separate post-hoc
        # pretrained discriminative baseline.
        "mcnemar_p_holm":
            np.nan,
        "mcnemar_significant_holm_0.05":
            np.nan,
    }

    combined = pd.concat(
        [
            existing,
            pd.DataFrame([cb_row]),
        ],
        ignore_index=True,
    )

    combined.to_csv(
        OUTDIR
        / "q1_with_codebert_table.csv",
        index=False,
    )

    print(
        json.dumps(
            {
                "n": len(merged),
                "primary_recall":
                    primary_recall,
                "codebert_recall":
                    codebert_recall,
                "paired":
                    {
                        **boot,
                        **mc,
                    },
            },
            indent=2,
        )
    )

    return {
        "n": int(len(merged)),
        "primary_recall":
            primary_recall,
        "codebert_recall":
            codebert_recall,
        "paired": {
            **boot,
            **mc,
        },
    }


def build_latex(
    internal,
    external,
    q1,
):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CodeBERT comparison with the primary sparse-text baseline.}",
        r"\label{tab:codebert_comparison}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & LR F1 & CodeBERT F1 & $\Delta$F1 & 95\% CI \\",
        r"\midrule",
    ]

    for name, obj in [
        ("Internal",
         internal),
        ("External",
         external),
    ]:
        lines.append(
            f"{name} & "
            f"{obj['primary']['f1']:.3f} & "
            f"{obj['codebert']['f1']:.3f} & "
            f"{obj['paired']['delta_f1']:+.3f} & "
            f"[{obj['paired']['delta_f1_ci_lo']:.3f}, "
            f"{obj['paired']['delta_f1_ci_hi']:.3f}] \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (
        OUTDIR
        / "table_codebert_internal_external.tex"
    ).write_text(
        "\n".join(lines) + "\n"
    )

    q1_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CodeBERT performance on the 200 lexically farthest external malicious scripts.}",
        r"\label{tab:q1_codebert}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Recall & $\Delta$Recall & 95\% CI \\",
        r"\midrule",
        (
            f"LR Char+Word & "
            f"{q1['primary_recall']:.3f} & "
            f"-- & -- \\\\"
        ),
        (
            f"CodeBERT & "
            f"{q1['codebert_recall']:.3f} & "
            f"{q1['paired']['delta_recall']:+.3f} & "
            f"[{q1['paired']['delta_recall_ci_lo']:.3f}, "
            f"{q1['paired']['delta_recall_ci_hi']:.3f}] \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (
        OUTDIR
        / "table_q1_codebert.tex"
    ).write_text(
        "\n".join(q1_lines) + "\n"
    )


def main():
    internal = analyze_internal()
    external = analyze_external()
    q1 = analyze_q1()

    summary = {
        "analysis_type":
            "secondary pretrained discriminative baseline analysis",
        "model":
            "microsoft/codebert-base",
        "n_bootstrap":
            N_BOOT,
        "bootstrap_seed":
            SEED,
        "internal":
            internal,
        "external_2000":
            external,
        "q1_farthest_malicious":
            q1,
        "multiple_testing_note":
            (
                "CodeBERT was added as a separate "
                "pretrained discriminative baseline. "
                "Previously frozen Holm-adjusted "
                "families for conventional classifiers "
                "and LLMs were not retroactively changed."
            ),
    }

    with open(
        OUTDIR
        / "codebert_statistics_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    build_latex(
        internal,
        external,
        q1,
    )

    print(
        "\n=== DONE ==="
    )

    print(
        "Outputs written to:",
        OUTDIR,
    )


if __name__ == "__main__":
    main()
