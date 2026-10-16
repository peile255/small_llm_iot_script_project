from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(".")
TRAIN_CSV = ROOT / "data/processed/final_frozen/final_train.csv"
TEST_CSV = ROOT / "data/processed/final_frozen/final_test.csv"
EXT_CSV = ROOT / "outputs/paper_v3/frozen_external_llm_subset_v1/external_llm_subset_2000.csv"

OUT = ROOT / "outputs/paper_v4/distribution_shift_v1"
OUT.mkdir(parents=True, exist_ok=True)


ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "cp1252",
    "latin-1",
]


def read_text(path):
    path = Path(path)

    data = path.read_bytes()

    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass

    return data.decode("latin-1", errors="replace")


def load_texts(df, path_col="path"):
    texts = []

    for i, p in enumerate(df[path_col].astype(str)):
        path = ROOT / p

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file at row {i}: {path}"
            )

        texts.append(read_text(path))

        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{len(df)}")

    return texts


def nearest_similarity(
    X_query,
    X_train,
    chunk_size=100,
):
    """
    Return maximum cosine similarity to any frozen
    training sample.

    TfidfVectorizer uses L2 normalization, so sparse
    dot product equals cosine similarity.
    """
    result = np.empty(
        X_query.shape[0],
        dtype=np.float32,
    )

    for start in range(
        0,
        X_query.shape[0],
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            X_query.shape[0],
        )

        sims = X_query[start:end] @ X_train.T

        if sparse.issparse(sims):
            # scipy sparse max returns sparse/matrix-like result
            mx = sims.max(axis=1)
            mx = np.asarray(mx.toarray()).ravel()
        else:
            mx = np.asarray(
                sims.max(axis=1)
            ).ravel()

        result[start:end] = mx

        print(
            f"  similarity {end}/{X_query.shape[0]}"
        )

    return result


print("Loading manifests...")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
ext = pd.read_csv(EXT_CSV)

assert len(train) == 4322
assert len(test) == 1265
assert len(ext) == 2000

print("train:", train.shape)
print("test:", test.shape)
print("external:", ext.shape)


print("\nLoading frozen training text...")
train_text = load_texts(train)

print("\nLoading internal-test text...")
test_text = load_texts(test)

print("\nLoading frozen external-subset text...")
ext_text = load_texts(ext)


print("\nFitting training-only TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(4, 5),
    min_df=2,
    max_features=150000,
    sublinear_tf=True,
    norm="l2",
    dtype=np.float32,
)

X_train = vectorizer.fit_transform(train_text)

print("X_train:", X_train.shape)


print("\nTransforming internal test...")
X_test = vectorizer.transform(test_text)

print("X_test:", X_test.shape)


print("\nTransforming external subset...")
X_ext = vectorizer.transform(ext_text)

print("X_ext:", X_ext.shape)


print("\nComputing nearest TRAIN similarity: internal...")
test_sim = nearest_similarity(
    X_test,
    X_train,
)

print("\nComputing nearest TRAIN similarity: external...")
ext_sim = nearest_similarity(
    X_ext,
    X_train,
)


# ------------------------------------
# Save per-sample results
# ------------------------------------

internal_out = test.copy()
internal_out.insert(
    0,
    "internal_test_index",
    np.arange(len(internal_out)),
)
internal_out["nearest_train_similarity"] = test_sim
internal_out["distribution"] = "internal_test"

external_out = ext.copy()
external_out["nearest_train_similarity"] = ext_sim
external_out["distribution"] = "external_paired"


internal_out.to_csv(
    OUT / "internal_test_train_similarity.csv",
    index=False,
)

external_out.to_csv(
    OUT / "external_2000_train_similarity.csv",
    index=False,
)


combined = pd.concat(
    [
        internal_out[
            [
                "internal_test_index",
                "label",
                "nearest_train_similarity",
                "distribution",
            ]
        ].rename(
            columns={
                "internal_test_index": "sample_index"
            }
        ),
        external_out[
            [
                "external_llm_index",
                "label",
                "nearest_train_similarity",
                "distribution",
            ]
        ].rename(
            columns={
                "external_llm_index": "sample_index"
            }
        ),
    ],
    ignore_index=True,
)

combined.to_csv(
    OUT / "sample_train_similarity.csv",
    index=False,
)


# ------------------------------------
# Distribution summaries
# ------------------------------------

def summary(x):
    x = np.asarray(x)

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "max": float(np.max(x)),
    }


summary_rows = []

for name, values in [
    ("internal_test", test_sim),
    ("external_paired", ext_sim),
]:
    row = {
        "distribution": name,
        **summary(values),
    }
    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUT / "similarity_summary.csv",
    index=False,
)


# ------------------------------------
# Distribution tests
# ------------------------------------

ks = ks_2samp(
    test_sim,
    ext_sim,
    alternative="two-sided",
    method="auto",
)

mw = mannwhitneyu(
    test_sim,
    ext_sim,
    alternative="two-sided",
)


tests = {
    "ks_statistic": float(ks.statistic),
    "ks_pvalue": float(ks.pvalue),
    "mannwhitney_u": float(mw.statistic),
    "mannwhitney_pvalue": float(mw.pvalue),
}


# Effect-size style descriptive differences
tests["mean_difference_external_minus_internal"] = (
    float(np.mean(ext_sim) - np.mean(test_sim))
)

tests["median_difference_external_minus_internal"] = (
    float(np.median(ext_sim) - np.median(test_sim))
)


# ------------------------------------
# Quantile bins based on EXTERNAL
# ------------------------------------

external_out["similarity_quintile"] = pd.qcut(
    external_out["nearest_train_similarity"],
    q=5,
    labels=[
        "Q1_farthest",
        "Q2",
        "Q3",
        "Q4",
        "Q5_closest",
    ],
    duplicates="drop",
)


quantile_summary = (
    external_out
    .groupby(
        "similarity_quintile",
        observed=False,
    )
    .agg(
        n=("label", "size"),
        malicious=("label", "sum"),
        similarity_min=(
            "nearest_train_similarity",
            "min",
        ),
        similarity_mean=(
            "nearest_train_similarity",
            "mean",
        ),
        similarity_median=(
            "nearest_train_similarity",
            "median",
        ),
        similarity_max=(
            "nearest_train_similarity",
            "max",
        ),
    )
    .reset_index()
)

quantile_summary["benign"] = (
    quantile_summary["n"]
    - quantile_summary["malicious"]
)

quantile_summary.to_csv(
    OUT / "external_similarity_quintiles.csv",
    index=False,
)


external_out.to_csv(
    OUT / "external_2000_similarity_binned.csv",
    index=False,
)


# ------------------------------------
# Fixed bins
# ------------------------------------

fixed_edges = [
    -np.inf,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    np.inf,
]

fixed_labels = [
    "<0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    ">=0.80",
]

external_out["similarity_fixed_bin"] = pd.cut(
    external_out["nearest_train_similarity"],
    bins=fixed_edges,
    labels=fixed_labels,
    right=False,
)


fixed_summary = (
    external_out
    .groupby(
        "similarity_fixed_bin",
        observed=False,
    )
    .agg(
        n=("label", "size"),
        malicious=("label", "sum"),
        similarity_mean=(
            "nearest_train_similarity",
            "mean",
        ),
    )
    .reset_index()
)

fixed_summary["benign"] = (
    fixed_summary["n"]
    - fixed_summary["malicious"]
)

fixed_summary.to_csv(
    OUT / "external_similarity_fixed_bins.csv",
    index=False,
)


# ------------------------------------
# JSON summary
# ------------------------------------

result = {
    "feature_space": {
        "fit_on": "frozen_train_only",
        "analyzer": "char_wb",
        "ngram_range": [4, 5],
        "min_df": 2,
        "max_features": 150000,
        "sublinear_tf": True,
        "norm": "l2",
        "n_features": int(X_train.shape[1]),
    },
    "internal_test": summary(test_sim),
    "external_paired": summary(ext_sim),
    "distribution_tests": tests,
}

with open(
    OUT / "distribution_shift_summary.json",
    "w",
) as f:
    json.dump(
        result,
        f,
        indent=2,
    )


print("\n=== SUMMARY ===")
print(summary_df.to_string(index=False))

print("\nKS:")
print(
    "D =",
    ks.statistic,
    "p =",
    ks.pvalue,
)

print("\nMann-Whitney:")
print(
    "U =",
    mw.statistic,
    "p =",
    mw.pvalue,
)

print("\nExternal quintiles:")
print(
    quantile_summary.to_string(
        index=False
    )
)

print("\nOutputs:")
for p in sorted(OUT.iterdir()):
    print(" ", p)
