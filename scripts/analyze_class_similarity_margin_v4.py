from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.stats import (
    ks_2samp,
    spearmanr,
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)


ROOT = Path(".")

TRAIN_CSV = (
    ROOT
    / "data/processed/final_frozen/final_train.csv"
)

EXT_CSV = (
    ROOT
    / "outputs/paper_v3/"
      "frozen_external_llm_subset_v1/"
      "external_llm_subset_2000.csv"
)

PAIRED_CSV = (
    ROOT
    / "outputs/paper_v3/"
      "frozen_external_results_v1/"
      "external_statistics_v3/"
      "paired_external_2000_predictions.csv"
)

OUT = (
    ROOT
    / "outputs/paper_v4/"
      "distribution_shift_v1/"
      "class_similarity_margin"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_COLS = {
    "LR Char+Word":
        "pred_lr_char_word",
    "LR Char":
        "pred_lr_char",
    "SVM Char":
        "pred_svm_char",
    "SVM Word":
        "pred_svm_word",
    "Mistral-7B":
        "pred_mistral_7b",
    "Phi-4":
        "pred_phi4",
    "Gemma-3-27B":
        "pred_gemma3_27b",
    "QwQ-32B":
        "pred_qwq_32b",
    "DeepSeek-R1-32B":
        "pred_deepseek_r1_32b",
}


ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "cp1252",
    "latin-1",
]


def read_text(path):

    data = Path(path).read_bytes()

    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass

    return data.decode(
        "latin-1",
        errors="replace",
    )


def load_texts(df):

    out = []

    for i, p in enumerate(
        df["path"].astype(str)
    ):
        out.append(
            read_text(ROOT / p)
        )

        if (i + 1) % 500 == 0:
            print(
                f"loaded {i+1}/{len(df)}"
            )

    return out


def row_max(
    Xq,
    Xt,
    chunk_size=100,
):

    result = np.empty(
        Xq.shape[0],
        dtype=np.float32,
    )

    for start in range(
        0,
        Xq.shape[0],
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            Xq.shape[0],
        )

        sims = (
            Xq[start:end]
            @ Xt.T
        )

        mx = (
            sims
            .max(axis=1)
            .toarray()
            .ravel()
        )

        result[start:end] = mx

        print(
            f"similarity {end}/{Xq.shape[0]}"
        )

    return result


print("Loading manifests...")

train = pd.read_csv(
    TRAIN_CSV
)

ext = pd.read_csv(
    EXT_CSV
)

pred = pd.read_csv(
    PAIRED_CSV
)

assert len(train) == 4322
assert len(ext) == 2000
assert len(pred) == 2000


print("Loading text...")

train_text = load_texts(train)
ext_text = load_texts(ext)


print("Fitting TRAIN-only TF-IDF...")

vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(4, 5),
    min_df=2,
    max_features=150000,
    sublinear_tf=True,
    norm="l2",
    dtype=np.float32,
)

X_train = vec.fit_transform(
    train_text
)

X_ext = vec.transform(
    ext_text
)


train_y = (
    train["label"]
    .astype(int)
    .to_numpy()
)

X_train_benign = X_train[
    train_y == 0
]

X_train_malicious = X_train[
    train_y == 1
]


print(
    "train benign:",
    X_train_benign.shape,
)

print(
    "train malicious:",
    X_train_malicious.shape,
)


print(
    "Nearest BENIGN train..."
)

sim_benign = row_max(
    X_ext,
    X_train_benign,
)


print(
    "Nearest MALICIOUS train..."
)

sim_malicious = row_max(
    X_ext,
    X_train_malicious,
)


result = ext[
    [
        "external_llm_index",
        "external_final_index",
        "label",
        "path",
        "normalized_sha256",
    ]
].copy()

result["sim_train_benign"] = (
    sim_benign
)

result["sim_train_malicious"] = (
    sim_malicious
)


# ---------------------------------------
# same/opposite similarity and margin
# ---------------------------------------

result[
    "same_class_similarity"
] = np.where(
    result["label"] == 0,
    result["sim_train_benign"],
    result["sim_train_malicious"],
)

result[
    "opposite_class_similarity"
] = np.where(
    result["label"] == 0,
    result["sim_train_malicious"],
    result["sim_train_benign"],
)

result[
    "class_similarity_margin"
] = (
    result["same_class_similarity"]
    -
    result["opposite_class_similarity"]
)


result[
    "nearest_train_predicted_class"
] = np.where(
    result["sim_train_malicious"]
    >
    result["sim_train_benign"],
    1,
    0,
)


result[
    "nearest_class_correct"
] = (
    result[
        "nearest_train_predicted_class"
    ]
    ==
    result["label"]
)


# ---------------------------------------
# merge predictions
# ---------------------------------------

merged = result.merge(
    pred,
    on=[
        "external_llm_index",
        "external_final_index",
    ],
    how="inner",
    suffixes=(
        "_sim",
        "_pred",
    ),
    validate="one_to_one",
)

assert len(merged) == 2000

assert np.array_equal(
    merged["label"].to_numpy(),
    merged["y_true"].to_numpy(),
)


# ---------------------------------------
# margin quintiles WITHIN CLASS
# ---------------------------------------

merged[
    "margin_quintile"
] = None

for label in [0, 1]:

    mask = (
        merged["y_true"]
        == label
    )

    q = pd.qcut(
        merged.loc[
            mask,
            "class_similarity_margin",
        ],
        q=5,
        labels=[
            "Q1_lowest_margin",
            "Q2",
            "Q3",
            "Q4",
            "Q5_highest_margin",
        ],
        duplicates="drop",
    )

    merged.loc[
        mask,
        "margin_quintile",
    ] = q.astype(str).values


# ---------------------------------------
# recall / specificity by margin
# ---------------------------------------

rows = []

for label, class_name in [
    (0, "benign"),
    (1, "malicious"),
]:

    dfc = merged[
        merged["y_true"] == label
    ]

    for q, g in dfc.groupby(
        "margin_quintile",
        sort=False,
    ):

        for model, col in (
            MODEL_COLS.items()
        ):

            pred_y = (
                g[col]
                .to_numpy()
            )

            correct = (
                pred_y == label
            )

            rows.append(
                {
                    "class":
                        class_name,
                    "label":
                        label,
                    "margin_quintile":
                        q,
                    "n":
                        len(g),
                    "margin_min":
                        g[
                            "class_similarity_margin"
                        ].min(),
                    "margin_mean":
                        g[
                            "class_similarity_margin"
                        ].mean(),
                    "margin_median":
                        g[
                            "class_similarity_margin"
                        ].median(),
                    "margin_max":
                        g[
                            "class_similarity_margin"
                        ].max(),
                    "model":
                        model,
                    "correct_rate":
                        correct.mean(),
                    "error_rate":
                        1 - correct.mean(),
                    "n_errors":
                        int(
                            (~correct).sum()
                        ),
                }
            )


perf = pd.DataFrame(rows)

perf.to_csv(
    OUT
    / "performance_by_class_margin_quintile.csv",
    index=False,
)


# ---------------------------------------
# continuous margin-error association
# ---------------------------------------

corr_rows = []

for label, class_name in [
    (0, "benign"),
    (1, "malicious"),
]:

    dfc = merged[
        merged["y_true"] == label
    ]

    margin = (
        dfc[
            "class_similarity_margin"
        ]
        .to_numpy()
    )

    for model, col in (
        MODEL_COLS.items()
    ):

        error = (
            dfc[col].to_numpy()
            != label
        ).astype(int)

        if np.unique(error).size < 2:
            rho = np.nan
            p = np.nan

        else:
            r = spearmanr(
                margin,
                error,
            )

            rho = r.statistic
            p = r.pvalue

        corr_rows.append(
            {
                "class":
                    class_name,
                "model":
                    model,
                "n":
                    len(dfc),
                "errors":
                    int(
                        error.sum()
                    ),
                "rho_margin_vs_error":
                    rho,
                "p_value":
                    p,
            }
        )


corr = pd.DataFrame(
    corr_rows
)

corr.to_csv(
    OUT
    / "margin_error_spearman_by_class.csv",
    index=False,
)


# ---------------------------------------
# Margin sign analysis
# ---------------------------------------

sign_rows = []

for label, class_name in [
    (0, "benign"),
    (1, "malicious"),
]:

    dfc = merged[
        merged["y_true"] == label
    ]

    for sign_name, mask in [
        (
            "margin_negative",
            dfc[
                "class_similarity_margin"
            ] < 0,
        ),
        (
            "margin_nonnegative",
            dfc[
                "class_similarity_margin"
            ] >= 0,
        ),
    ]:

        g = dfc[mask]

        for model, col in (
            MODEL_COLS.items()
        ):

            if len(g) == 0:
                continue

            correct = (
                g[col].to_numpy()
                == label
            )

            sign_rows.append(
                {
                    "class":
                        class_name,
                    "margin_group":
                        sign_name,
                    "model":
                        model,
                    "n":
                        len(g),
                    "correct_rate":
                        correct.mean(),
                    "error_rate":
                        1
                        - correct.mean(),
                }
            )


sign_df = pd.DataFrame(
    sign_rows
)

sign_df.to_csv(
    OUT
    / "performance_by_margin_sign.csv",
    index=False,
)


# ---------------------------------------
# basic summary
# ---------------------------------------

summary_rows = []

for label, class_name in [
    (0, "benign"),
    (1, "malicious"),
]:

    g = merged[
        merged["y_true"] == label
    ]

    summary_rows.append(
        {
            "class":
                class_name,
            "n":
                len(g),
            "mean_same_similarity":
                g[
                    "same_class_similarity"
                ].mean(),
            "mean_opposite_similarity":
                g[
                    "opposite_class_similarity"
                ].mean(),
            "mean_margin":
                g[
                    "class_similarity_margin"
                ].mean(),
            "median_margin":
                g[
                    "class_similarity_margin"
                ].median(),
            "negative_margin_n":
                int(
                    (
                        g[
                            "class_similarity_margin"
                        ]
                        < 0
                    ).sum()
                ),
            "negative_margin_rate":
                (
                    g[
                        "class_similarity_margin"
                    ]
                    < 0
                ).mean(),
            "nearest_class_accuracy":
                g[
                    "nearest_class_correct"
                ].mean(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT
    / "class_similarity_margin_summary.csv",
    index=False,
)


merged.to_csv(
    OUT
    / "paired_external_2000_class_similarity_margin.csv",
    index=False,
)


with open(
    OUT / "analysis_summary.json",
    "w",
) as f:

    json.dump(
        {
            "n": 2000,
            "feature_space":
                "train-fitted char_wb TF-IDF 4-5",
            "margin_definition":
                "nearest same-class similarity minus nearest opposite-class similarity",
        },
        f,
        indent=2,
    )


print("\n=== CLASS MARGIN SUMMARY ===")
print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\n=== PERFORMANCE BY MARGIN SIGN ==="
)

print(
    sign_df.to_string(
        index=False
    )
)


print(
    "\n=== MARGIN VS ERROR SPEARMAN ==="
)

print(
    corr.to_string(
        index=False
    )
)


print(
    "\n=== PERFORMANCE BY MARGIN QUINTILE ==="
)

print(
    perf.to_string(
        index=False
    )
)


print("\nOutputs:")

for p in sorted(
    OUT.iterdir()
):
    print(
        " ",
        p,
    )
