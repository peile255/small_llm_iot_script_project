import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors


RAW_ROOT = Path("data/raw/mpsd")
OUT_DIR = Path("data/processed")

BENIGN_DIR = RAW_ROOT / "powershell_benign_dataset"
MALICIOUS_DIR = RAW_ROOT / "malicious_pure"

RANDOM_STATE = 20260911
NEAR_DUP_SIMILARITY = 0.90


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def decode_bytes(data):
    encodings = [
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "cp1252",
        "latin-1",
    ]

    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeError:
            pass

    return data.decode("utf-8", errors="replace")


def normalize_for_similarity(text):
    text = text.lower()

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse all whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def collect_files(directory, label, subset):
    records = []

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file()
    )

    for path in files:
        data = path.read_bytes()

        text = decode_bytes(data)
        normalized = normalize_for_similarity(text)

        records.append(
            {
                "path": str(path),
                "sha256": sha256_bytes(data),
                "normalized_sha256": sha256_bytes(
                    normalized.encode(
                        "utf-8",
                        errors="replace",
                    )
                ),
                "label": int(label),
                "source": "mpsd",
                "subset": subset,
                "n_bytes": len(data),
                "_normalized_text": normalized,
            }
        )

    return records


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[
                self.parent[x]
            ]
            x = self.parent[x]

        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def load_development_hashes():
    """
    Any group containing a sample previously used during
    development is forbidden from final_test.
    """

    paths = [
        OUT_DIR / "test.csv",
        OUT_DIR / "val_smoke100.csv",
        OUT_DIR / "gateway_smoke10.csv",
    ]

    hashes = set()

    for path in paths:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if "sha256" not in df.columns:
            continue

        hashes.update(
            df["sha256"]
            .dropna()
            .astype(str)
            .tolist()
        )

        print(
            f"Development hashes from {path}: "
            f"{len(df)} rows"
        )

    return hashes


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Collecting MPSD primary samples...")

    records = []

    records.extend(
        collect_files(
            BENIGN_DIR,
            label=0,
            subset="benign",
        )
    )

    records.extend(
        collect_files(
            MALICIOUS_DIR,
            label=1,
            subset="malicious_pure",
        )
    )

    df = pd.DataFrame(records)

    print("\nRaw primary samples:", len(df))
    print(
        "Raw class counts:",
        df["label"].value_counts().sort_index().to_dict(),
    )

    # -------------------------------------------------
    # Detect exact SHA conflicts across labels.
    # -------------------------------------------------

    hash_label_counts = (
        df.groupby("sha256")["label"]
        .nunique()
    )

    conflicting_hashes = set(
        hash_label_counts[
            hash_label_counts > 1
        ].index
    )

    print(
        "Cross-label exact-hash conflicts:",
        len(conflicting_hashes),
    )

    if conflicting_hashes:
        df = df[
            ~df["sha256"].isin(
                conflicting_hashes
            )
        ].copy()

    # Exact dedup within same label.
    before_exact = len(df)

    df = (
        df.sort_values(
            ["label", "path"]
        )
        .drop_duplicates(
            subset=["sha256"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        "Exact duplicate rows removed:",
        before_exact - len(df),
    )

    print(
        "After exact dedup:",
        len(df),
    )

    # -------------------------------------------------
    # Near-duplicate grouping.
    # -------------------------------------------------

    print("\nBuilding TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 5),
        lowercase=False,
        min_df=2,
        max_features=150000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(
        df["_normalized_text"]
    )

    print("TF-IDF shape:", X.shape)

    uf = UnionFind(len(df))

    # Always union whitespace/case-normalized duplicates.
    for _, idxs in (
        df.groupby("normalized_sha256")
        .groups
        .items()
    ):
        idxs = list(idxs)

        if len(idxs) > 1:
            first = idxs[0]

            for other in idxs[1:]:
                uf.union(first, other)

    print(
        "Finding near duplicates with cosine "
        f"similarity >= {NEAR_DUP_SIMILARITY:.2f}..."
    )

    nn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(X)

    graph = nn.radius_neighbors_graph(
        X,
        radius=1.0 - NEAR_DUP_SIMILARITY,
        mode="connectivity",
        sort_results=False,
    )

    graph = graph.tocoo()

    edge_count = 0

    for a, b in zip(
        graph.row,
        graph.col,
    ):
        if a >= b:
            continue

        uf.union(int(a), int(b))
        edge_count += 1

    print("Near-duplicate edges:", edge_count)

    roots = [
        uf.find(i)
        for i in range(len(df))
    ]

    root_to_group = {}

    group_ids = []

    for root in roots:
        if root not in root_to_group:
            root_to_group[root] = (
                len(root_to_group)
            )

        group_ids.append(
            root_to_group[root]
        )

    df["group_id"] = group_ids

    print(
        "Near-duplicate groups:",
        df["group_id"].nunique(),
    )

    group_sizes = (
        df.groupby("group_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        "Largest group size:",
        int(group_sizes.iloc[0]),
    )

    print(
        "Groups with >1 sample:",
        int((group_sizes > 1).sum()),
    )

    # -------------------------------------------------
    # Protect development-contaminated groups.
    # -------------------------------------------------

    development_hashes = (
        load_development_hashes()
    )

    development_rows = df[
        df["sha256"].isin(
            development_hashes
        )
    ]

    tainted_groups = set(
        development_rows[
            "group_id"
        ].tolist()
    )

    df["development_tainted_group"] = (
        df["group_id"].isin(
            tainted_groups
        )
    )

    print(
        "\nDevelopment hashes found in primary pool:",
        len(development_rows),
    )

    print(
        "Development-tainted groups:",
        len(tainted_groups),
    )

    print(
        "Rows in development-tainted groups:",
        int(
            df[
                "development_tainted_group"
            ].sum()
        ),
    )

    # -------------------------------------------------
    # Optimized group-aware final split.
    #
    # We search across many StratifiedGroupKFold
    # candidates instead of taking the first fold.
    # This is important because some near-duplicate
    # groups are very large and can distort class ratios.
    # -------------------------------------------------

    total_n = len(df)
    overall_positive_rate = float(df["label"].mean())

    target_test_n = round(total_n * 0.20)
    target_val_n = round(total_n * 0.10)

    print("\nOverall positive rate:", overall_positive_rate)
    print("Target test n:", target_test_n)
    print("Target val n:", target_val_n)

    def choose_best_fold(
        frame,
        n_splits,
        target_n,
        target_positive_rate,
        seed_start,
        n_seeds=100,
    ):
        best = None

        for seed_offset in range(n_seeds):
            seed = seed_start + seed_offset

            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )

            for fold_id, (_, candidate_idx) in enumerate(
                splitter.split(
                    frame,
                    frame["label"],
                    groups=frame["group_id"],
                )
            ):
                candidate = frame.iloc[candidate_idx]

                n = len(candidate)

                if n == 0:
                    continue

                positive_rate = float(
                    candidate["label"].mean()
                )

                # Relative size error.
                size_error = abs(
                    n - target_n
                ) / max(target_n, 1)

                # Absolute prevalence error.
                prevalence_error = abs(
                    positive_rate
                    - target_positive_rate
                )

                # Prioritize class-distribution fidelity,
                # while also matching target split size.
                score = (
                    2.0 * prevalence_error
                    + size_error
                )

                record = {
                    "score": score,
                    "seed": seed,
                    "fold_id": fold_id,
                    "n": n,
                    "positive_rate": positive_rate,
                    "idx": candidate_idx,
                }

                if (
                    best is None
                    or record["score"] < best["score"]
                ):
                    best = record

        if best is None:
            raise RuntimeError(
                "Unable to find candidate group split."
            )

        print(
            "Selected candidate:",
            {
                "seed": best["seed"],
                "fold_id": best["fold_id"],
                "n": best["n"],
                "positive_rate":
                    best["positive_rate"],
                "score": best["score"],
            },
        )

        return best["idx"]

    # -------------------------------------------------
    # FINAL TEST
    #
    # Test may only contain clean groups.
    #
    # Clean pool is smaller than the complete dataset,
    # therefore use approximately 3-fold candidates
    # and select the candidate closest to 20% of the
    # complete dataset with the global class ratio.
    # -------------------------------------------------

    clean = df[
        ~df["development_tainted_group"]
    ].copy().reset_index(drop=True)

    final_test_idx = choose_best_fold(
        frame=clean,
        n_splits=3,
        target_n=target_test_n,
        target_positive_rate=overall_positive_rate,
        seed_start=RANDOM_STATE,
        n_seeds=150,
    )

    final_test = (
        clean.iloc[final_test_idx]
        .copy()
    )

    final_test_groups = set(
        final_test["group_id"]
    )

    trainval = df[
        ~df["group_id"].isin(
            final_test_groups
        )
    ].copy().reset_index(drop=True)

    # -------------------------------------------------
    # FINAL VALIDATION
    #
    # Select approximately 10% of the complete dataset.
    # Search many group-aware candidates and preserve
    # overall class prevalence as closely as possible.
    # -------------------------------------------------

    remaining_fraction = (
        target_val_n / len(trainval)
    )

    approximate_val_splits = max(
        2,
        round(1.0 / remaining_fraction),
    )

    print(
        "Validation SGKF n_splits:",
        approximate_val_splits,
    )

    final_val_idx = choose_best_fold(
        frame=trainval,
        n_splits=approximate_val_splits,
        target_n=target_val_n,
        target_positive_rate=overall_positive_rate,
        seed_start=RANDOM_STATE + 1000,
        n_seeds=150,
    )

    final_val = (
        trainval.iloc[final_val_idx]
        .copy()
    )

    final_val_groups = set(
        final_val["group_id"]
    )

    final_train = trainval[
        ~trainval["group_id"].isin(
            final_val_groups
        )
    ].copy()

    # -------------------------------------------------
    # Safety checks.
    # -------------------------------------------------

    split_frames = {
        "train": final_train,
        "val": final_val,
        "test": final_test,
    }

    for name, frame in split_frames.items():
        print(
            f"\n{name}: n={len(frame)} "
            f"class_counts="
            f"{frame['label'].value_counts().sort_index().to_dict()}"
        )

    # SHA separation.
    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        sha_overlap = (
            set(split_frames[a]["sha256"])
            & set(split_frames[b]["sha256"])
        )

        group_overlap = (
            set(split_frames[a]["group_id"])
            & set(split_frames[b]["group_id"])
        )

        print(
            f"{a}/{b} SHA overlap:",
            len(sha_overlap),
        )

        print(
            f"{a}/{b} group overlap:",
            len(group_overlap),
        )

        assert not sha_overlap
        assert not group_overlap

    # No development-tainted group may enter final test.
    assert not final_test[
        "development_tainted_group"
    ].any()

    # No known development hash may enter final test.
    dev_hash_test_overlap = (
        set(final_test["sha256"])
        & development_hashes
    )

    print(
        "\nDevelopment hash overlap with final_test:",
        len(dev_hash_test_overlap),
    )

    assert not dev_hash_test_overlap

    # -------------------------------------------------
    # Save.
    # -------------------------------------------------

    export_cols = [
        "path",
        "sha256",
        "label",
        "source",
        "subset",
        "n_bytes",
        "group_id",
        "development_tainted_group",
    ]

    final_train[export_cols].to_csv(
        OUT_DIR / "final_train.csv",
        index=False,
    )

    final_val[export_cols].to_csv(
        OUT_DIR / "final_val.csv",
        index=False,
    )

    final_test[export_cols].to_csv(
        OUT_DIR / "final_test.csv",
        index=False,
    )

    summary = {
        "random_state": RANDOM_STATE,
        "near_duplicate_similarity_threshold":
            NEAR_DUP_SIMILARITY,
        "raw_primary_n": len(records),
        "cross_label_hash_conflicts":
            len(conflicting_hashes),
        "deduplicated_primary_n": len(df),
        "near_duplicate_groups":
            int(df["group_id"].nunique()),
        "near_duplicate_edges":
            int(edge_count),
        "development_tainted_groups":
            len(tainted_groups),
        "train": {
            "n": len(final_train),
            "class_counts":
                final_train[
                    "label"
                ].value_counts().sort_index().to_dict(),
        },
        "val": {
            "n": len(final_val),
            "class_counts":
                final_val[
                    "label"
                ].value_counts().sort_index().to_dict(),
        },
        "test": {
            "n": len(final_test),
            "class_counts":
                final_test[
                    "label"
                ].value_counts().sort_index().to_dict(),
        },
    }

    with open(
        OUT_DIR / "final_split_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\nSaved:",
        OUT_DIR / "final_train.csv",
    )
    print(
        "Saved:",
        OUT_DIR / "final_val.csv",
    )
    print(
        "Saved:",
        OUT_DIR / "final_test.csv",
    )
    print(
        "Saved:",
        OUT_DIR / "final_split_summary.json",
    )

    print(
        "\nFINAL SPLIT VALIDATION PASSED."
    )


if __name__ == "__main__":
    main()
