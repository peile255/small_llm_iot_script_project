#!/usr/bin/env python3

from pathlib import Path
import json
import re
import tarfile

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


THRESHOLD = 0.90

ARCHIVE = Path(
    "data/external/wwe123/ast_ps1_by_label.tar.gz"
)

FROZEN = Path(
    "data/processed/final_frozen"
)

EXTERNAL_CLEAN = Path(
    "outputs/paper_v3/external_clean/external_exact_clean.csv"
)

OUT_DIR = Path(
    "outputs/paper_v3/external_neardup"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# EXACT preprocessing alignment
# --------------------------------------------------

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

    return data.decode(
        "utf-8",
        errors="replace",
    )


def normalize_for_similarity(text):
    text = text.lower()

    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# --------------------------------------------------
# Load deduplicated frozen primary universe
# --------------------------------------------------

def load_primary():

    frames = []

    for name in [
        "final_train.csv",
        "final_val.csv",
        "final_test.csv",
    ]:
        path = FROZEN / name
        df = pd.read_csv(path)
        frames.append(df)

    primary = pd.concat(
        frames,
        ignore_index=True,
    )

    assert len(primary) == 6196, (
        f"Expected 6196 frozen primary rows, "
        f"got {len(primary)}"
    )

    assert primary["sha256"].nunique() == 6196

    texts = []

    for i, row in primary.iterrows():

        path = Path(row["path"])

        raw = path.read_bytes()

        text = normalize_for_similarity(
            decode_bytes(raw)
        )

        texts.append(text)

        if (i + 1) % 1000 == 0:
            print(
                f"Primary loaded: {i + 1}/6196",
                flush=True,
            )

    primary = primary.copy()
    primary["_text"] = texts

    return primary


# --------------------------------------------------
# Load exact-clean external texts from tar
# --------------------------------------------------

def load_external():

    clean = pd.read_csv(
        EXTERNAL_CLEAN
    )

    wanted = set(
        clean["archive_path"]
        .astype(str)
        .tolist()
    )

    text_map = {}

    count = 0

    with tarfile.open(
        ARCHIVE,
        mode="r|gz",
    ) as tf:

        for member in tf:

            if not member.isfile():
                continue

            if member.name not in wanted:
                continue

            f = tf.extractfile(member)

            if f is None:
                continue

            raw = f.read()

            text_map[member.name] = (
                normalize_for_similarity(
                    decode_bytes(raw)
                )
            )

            count += 1

            if count % 1000 == 0:
                print(
                    f"External loaded: "
                    f"{count}/{len(wanted)}",
                    flush=True,
                )

    missing = wanted - set(text_map)

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} "
            f"external archive members"
        )

    clean = clean.copy()

    clean["_text"] = (
        clean["archive_path"]
        .map(text_map)
    )

    assert clean["_text"].notna().all()

    return clean


def main():

    print(
        "Loading frozen primary...",
        flush=True,
    )

    primary = load_primary()

    print(
        "\nLoading exact-clean external...",
        flush=True,
    )

    external = load_external()

    print(
        "\nPrimary rows:",
        len(primary),
    )

    print(
        "External exact-clean rows:",
        len(external),
    )

    # --------------------------------------------------
    # Fit TF-IDF on frozen deduplicated primary only.
    #
    # This mirrors the primary grouping feature space
    # and treats the cross-corpus screen conservatively.
    # --------------------------------------------------

    print(
        "\nFitting primary TF-IDF...",
        flush=True,
    )

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

    X_primary = vectorizer.fit_transform(
        primary["_text"]
    )

    X_external = vectorizer.transform(
        external["_text"]
    )

    print(
        "Primary TF-IDF shape:",
        X_primary.shape,
        flush=True,
    )

    print(
        "External TF-IDF shape:",
        X_external.shape,
        flush=True,
    )

    # --------------------------------------------------
    # Nearest primary neighbour for each external row
    # --------------------------------------------------

    print(
        "\nFinding nearest primary neighbours...",
        flush=True,
    )

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(X_primary)

    distances, indices = nn.kneighbors(
        X_external,
        return_distance=True,
    )

    distances = distances[:, 0]
    indices = indices[:, 0]

    similarities = 1.0 - distances

    result = external.drop(
        columns=["_text"]
    ).copy()

    result[
        "nearest_primary_similarity"
    ] = similarities

    result[
        "nearest_primary_path"
    ] = (
        primary.iloc[indices]["path"]
        .to_numpy()
    )

    result[
        "nearest_primary_sha256"
    ] = (
        primary.iloc[indices]["sha256"]
        .to_numpy()
    )

    result[
        "nearest_primary_label"
    ] = (
        primary.iloc[indices]["label"]
        .to_numpy()
    )

    if "group_id" in primary.columns:
        result[
            "nearest_primary_group_id"
        ] = (
            primary.iloc[indices]["group_id"]
            .to_numpy()
        )

    result[
        "near_duplicate_primary"
    ] = (
        result[
            "nearest_primary_similarity"
        ] >= THRESHOLD
    )

    # --------------------------------------------------
    # Split removed vs kept
    # --------------------------------------------------

    removed = result[
        result["near_duplicate_primary"]
    ].copy()

    kept = result[
        ~result["near_duplicate_primary"]
    ].copy()

    result.to_csv(
        OUT_DIR /
        "external_nearest_primary.csv",
        index=False,
    )

    removed.to_csv(
        OUT_DIR /
        "removed_primary_neardup.csv",
        index=False,
    )

    kept = kept.sort_values(
        [
            "label",
            "archive_path",
        ]
    ).reset_index(drop=True)

    kept.insert(
        0,
        "external_final_index",
        range(len(kept)),
    )

    kept.to_csv(
        OUT_DIR /
        "external_final_audited.csv",
        index=False,
    )

    # --------------------------------------------------
    # Similarity distribution
    # --------------------------------------------------

    quantiles = {
        str(q): float(
            np.quantile(
                similarities,
                q,
            )
        )
        for q in [
            0.50,
            0.90,
            0.95,
            0.99,
            0.999,
            1.00,
        ]
    }

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "threshold":
            THRESHOLD,

        "vectorizer":
            "char_wb TF-IDF 4-5",

        "fit_scope":
            "frozen deduplicated primary only",

        "primary_rows":
            int(len(primary)),

        "external_exact_clean_rows":
            int(len(external)),

        "external_exact_clean_label_0":
            int(
                (external["label"] == 0)
                .sum()
            ),

        "external_exact_clean_label_1":
            int(
                (external["label"] == 1)
                .sum()
            ),

        "removed_neardup_rows":
            int(len(removed)),

        "removed_neardup_label_0":
            int(
                (removed["label"] == 0)
                .sum()
            ),

        "removed_neardup_label_1":
            int(
                (removed["label"] == 1)
                .sum()
            ),

        "final_audited_rows":
            int(len(kept)),

        "final_audited_label_0":
            int(
                (kept["label"] == 0)
                .sum()
            ),

        "final_audited_label_1":
            int(
                (kept["label"] == 1)
                .sum()
            ),

        "same_label_neardup_rows":
            int(
                (
                    removed["label"]
                    ==
                    removed[
                        "nearest_primary_label"
                    ]
                ).sum()
            ),

        "cross_label_neardup_rows":
            int(
                (
                    removed["label"]
                    !=
                    removed[
                        "nearest_primary_label"
                    ]
                ).sum()
            ),

        "similarity_quantiles":
            quantiles,
    }

    (
        OUT_DIR /
        "neardup_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    print(
        "\n=== NEAR-DUP AUDIT SUMMARY ==="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nTop 20 similarities:"
    )

    cols = [
        "archive_path",
        "label",
        "nearest_primary_path",
        "nearest_primary_label",
        "nearest_primary_similarity",
    ]

    print(
        result
        .sort_values(
            "nearest_primary_similarity",
            ascending=False,
        )[cols]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
