#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import re
import tarfile

import pandas as pd


ARCHIVE = Path(
    "data/external/wwe123/ast_ps1_by_label.tar.gz"
)

RAW_ROOT = Path("data/raw/mpsd")

BENIGN_DIR = (
    RAW_ROOT / "powershell_benign_dataset"
)

MALICIOUS_DIR = (
    RAW_ROOT / "malicious_pure"
)

OUT_DIR = Path(
    "outputs/paper_v3/external_audit"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# EXACTLY aligned with prepare_final_split.py
# --------------------------------------------------

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
            return data.decode(enc), enc
        except UnicodeError:
            pass

    return (
        data.decode(
            "utf-8",
            errors="replace",
        ),
        "utf-8-replace",
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


def normalized_sha(data):
    text, encoding = decode_bytes(data)

    norm = normalize_for_similarity(text)

    sha = sha256_bytes(
        norm.encode(
            "utf-8",
            errors="replace",
        )
    )

    return text, encoding, sha


# --------------------------------------------------
# Labels
# --------------------------------------------------

def infer_external_label(path):
    parts = Path(path).parts

    if "ast_ps1_by_label" not in parts:
        return None

    i = parts.index(
        "ast_ps1_by_label"
    )

    if i + 1 >= len(parts):
        return None

    value = parts[i + 1]

    if value == "0":
        return 0

    if value == "1":
        return 1

    return None


# --------------------------------------------------
# External
# --------------------------------------------------

def build_external_manifest():

    rows = []
    count = 0

    with tarfile.open(
        ARCHIVE,
        mode="r|gz",
    ) as tf:

        for member in tf:

            if not member.isfile():
                continue

            if not member.name.lower().endswith(
                ".ps1"
            ):
                continue

            f = tf.extractfile(member)

            if f is None:
                continue

            raw = f.read()

            text, encoding, norm_sha = (
                normalized_sha(raw)
            )

            rows.append({
                "external_index": count,
                "archive_path":
                    member.name,
                "label":
                    infer_external_label(
                        member.name
                    ),
                "n_bytes":
                    len(raw),
                "n_chars":
                    len(text),
                "encoding":
                    encoding,
                "raw_sha256":
                    sha256_bytes(raw),
                "normalized_sha256":
                    norm_sha,
            })

            count += 1

            if count % 1000 == 0:
                print(
                    "External hashed:",
                    count,
                    flush=True,
                )

    return pd.DataFrame(rows)


# --------------------------------------------------
# Primary EXACT source corpus
# --------------------------------------------------

def collect_primary_dir(
    directory,
    label,
):

    rows = []

    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
    )

    for path in files:

        raw = path.read_bytes()

        text, encoding, norm_sha = (
            normalized_sha(raw)
        )

        rows.append({
            "primary_path":
                str(path),
            "label":
                int(label),
            "n_bytes":
                len(raw),
            "n_chars":
                len(text),
            "encoding":
                encoding,
            "raw_sha256":
                sha256_bytes(raw),
            "normalized_sha256":
                norm_sha,
        })

    return rows


def build_primary_manifest():

    rows = []

    rows.extend(
        collect_primary_dir(
            BENIGN_DIR,
            label=0,
        )
    )

    rows.extend(
        collect_primary_dir(
            MALICIOUS_DIR,
            label=1,
        )
    )

    return pd.DataFrame(rows)


def main():

    print(
        "Building aligned external manifest...",
        flush=True,
    )

    ext = build_external_manifest()

    print(
        "\nExternal total:",
        len(ext),
    )

    print(
        ext["label"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    assert len(ext) == 30595
    assert ext["label"].isna().sum() == 0

    ext.to_csv(
        OUT_DIR /
        "external_manifest_raw.csv",
        index=False,
    )

    # ------------------------------------------------
    # Primary
    # ------------------------------------------------

    print(
        "\nBuilding aligned primary reference...",
        flush=True,
    )

    primary = build_primary_manifest()

    print(
        "Primary reference files:",
        len(primary),
    )

    print(
        primary["label"]
        .value_counts()
        .sort_index()
    )

    # Critical provenance assertion.
    assert len(primary) == 8518, (
        "Expected exactly 8518 raw primary "
        f"samples, got {len(primary)}"
    )

    primary.to_csv(
        OUT_DIR /
        "primary_reference_manifest.csv",
        index=False,
    )

    # ------------------------------------------------
    # External internal duplicate / conflicts
    # ------------------------------------------------

    raw_dup_mask = ext.duplicated(
        "raw_sha256",
        keep=False,
    )

    norm_dup_mask = ext.duplicated(
        "normalized_sha256",
        keep=False,
    )

    ext[
        raw_dup_mask
    ].sort_values(
        "raw_sha256"
    ).to_csv(
        OUT_DIR /
        "external_internal_raw_duplicates.csv",
        index=False,
    )

    ext[
        norm_dup_mask
    ].sort_values(
        "normalized_sha256"
    ).to_csv(
        OUT_DIR /
        "external_internal_normalized_duplicates.csv",
        index=False,
    )

    # Cross-label conflicts
    raw_counts = (
        ext.groupby("raw_sha256")["label"]
        .nunique()
    )

    raw_conflict_hashes = set(
        raw_counts[
            raw_counts > 1
        ].index
    )

    raw_conflicts = ext[
        ext["raw_sha256"].isin(
            raw_conflict_hashes
        )
    ]

    raw_conflicts.to_csv(
        OUT_DIR /
        "external_cross_label_raw_conflicts.csv",
        index=False,
    )

    norm_counts = (
        ext.groupby(
            "normalized_sha256"
        )["label"]
        .nunique()
    )

    norm_conflict_hashes = set(
        norm_counts[
            norm_counts > 1
        ].index
    )

    norm_conflicts = ext[
        ext[
            "normalized_sha256"
        ].isin(
            norm_conflict_hashes
        )
    ]

    norm_conflicts.to_csv(
        OUT_DIR /
        "external_cross_label_normalized_conflicts.csv",
        index=False,
    )

    # ------------------------------------------------
    # Cross-corpus exact overlap
    # ------------------------------------------------

    raw_overlap = ext.merge(
        primary[
            [
                "primary_path",
                "label",
                "raw_sha256",
            ]
        ].rename(
            columns={
                "label":
                    "primary_label"
            }
        ),
        on="raw_sha256",
        how="inner",
    )

    norm_overlap = ext.merge(
        primary[
            [
                "primary_path",
                "label",
                "normalized_sha256",
            ]
        ].rename(
            columns={
                "label":
                    "primary_label"
            }
        ),
        on="normalized_sha256",
        how="inner",
    )

    raw_overlap.to_csv(
        OUT_DIR /
        "exact_raw_sha_overlap.csv",
        index=False,
    )

    norm_overlap.to_csv(
        OUT_DIR /
        "normalized_sha_overlap.csv",
        index=False,
    )

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    summary = {
        "preprocessing_alignment":
            "prepare_final_split.py",

        "external_total":
            int(len(ext)),

        "external_label_0":
            int(
                (ext["label"] == 0)
                .sum()
            ),

        "external_label_1":
            int(
                (ext["label"] == 1)
                .sum()
            ),

        "external_unknown_label":
            int(
                ext["label"]
                .isna()
                .sum()
            ),

        "external_unique_raw_sha":
            int(
                ext[
                    "raw_sha256"
                ].nunique()
            ),

        "external_unique_normalized_sha":
            int(
                ext[
                    "normalized_sha256"
                ].nunique()
            ),

        "external_internal_raw_duplicate_rows":
            int(raw_dup_mask.sum()),

        "external_internal_normalized_duplicate_rows":
            int(norm_dup_mask.sum()),

        "external_cross_label_raw_conflict_rows":
            int(len(raw_conflicts)),

        "external_cross_label_normalized_conflict_rows":
            int(len(norm_conflicts)),

        "primary_reference_files":
            int(len(primary)),

        "primary_label_0":
            int(
                (primary["label"] == 0)
                .sum()
            ),

        "primary_label_1":
            int(
                (primary["label"] == 1)
                .sum()
            ),

        "exact_raw_overlap_rows":
            int(len(raw_overlap)),

        "exact_raw_overlap_unique_external":
            int(
                raw_overlap[
                    "archive_path"
                ].nunique()
            ),

        "normalized_overlap_rows":
            int(len(norm_overlap)),

        "normalized_overlap_unique_external":
            int(
                norm_overlap[
                    "archive_path"
                ].nunique()
            ),
    }

    (
        OUT_DIR /
        "exact_audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    print(
        "\n=== ALIGNED EXACT AUDIT SUMMARY ==="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
