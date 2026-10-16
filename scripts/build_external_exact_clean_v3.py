#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd


AUDIT_DIR = Path(
    "outputs/paper_v3/external_audit"
)

OUT_DIR = Path(
    "outputs/paper_v3/external_clean"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = AUDIT_DIR / "external_manifest_raw.csv"
PRIMARY = AUDIT_DIR / "primary_reference_manifest.csv"


def main():
    ext = pd.read_csv(MANIFEST)
    primary = pd.read_csv(PRIMARY)

    print("Raw external rows:", len(ext))
    print("Raw label counts:")
    print(ext["label"].value_counts().sort_index())
    print()

    # -------------------------------------------------
    # 1. Identify cross-label normalized-hash conflicts
    # -------------------------------------------------
    norm_label_n = (
        ext.groupby("normalized_sha256")["label"]
        .nunique()
    )

    conflict_hashes = set(
        norm_label_n[norm_label_n > 1].index
    )

    conflict_rows = ext[
        ext["normalized_sha256"].isin(conflict_hashes)
    ].copy()

    conflict_rows.to_csv(
        OUT_DIR /
        "removed_cross_label_conflicts.csv",
        index=False,
    )

    print(
        "Cross-label conflict hashes:",
        len(conflict_hashes)
    )
    print(
        "Cross-label conflict rows:",
        len(conflict_rows)
    )

    # Remove all members of conflict hashes
    clean = ext[
        ~ext["normalized_sha256"].isin(
            conflict_hashes
        )
    ].copy()

    # -------------------------------------------------
    # 2. Internal normalized-SHA deduplication
    # -------------------------------------------------
    before_dedup = len(clean)

    # deterministic:
    # retain first row by archive_path order
    clean = (
        clean.sort_values(
            ["normalized_sha256", "archive_path"]
        )
        .drop_duplicates(
            "normalized_sha256",
            keep="first",
        )
        .copy()
    )

    removed_internal_n = before_dedup - len(clean)

    removed_internal = ext[
        ~ext["archive_path"].isin(
            set(clean["archive_path"])
            |
            set(conflict_rows["archive_path"])
        )
    ].copy()

    removed_internal.to_csv(
        OUT_DIR /
        "removed_internal_duplicates.csv",
        index=False,
    )

    print(
        "Removed internal duplicate copies:",
        removed_internal_n
    )
    print(
        "After conflict removal + dedup:",
        len(clean)
    )

    # -------------------------------------------------
    # 3. Remove all normalized hashes present
    #    anywhere in primary corpus
    # -------------------------------------------------
    primary_hashes = set(
        primary["normalized_sha256"]
        .dropna()
        .astype(str)
    )

    exact_overlap_mask = (
        clean["normalized_sha256"]
        .astype(str)
        .isin(primary_hashes)
    )

    removed_overlap = clean[
        exact_overlap_mask
    ].copy()

    kept = clean[
        ~exact_overlap_mask
    ].copy()

    removed_overlap.to_csv(
        OUT_DIR /
        "removed_primary_exact_overlap.csv",
        index=False,
    )

    kept = kept.sort_values(
        ["label", "archive_path"]
    ).reset_index(drop=True)

    kept.insert(
        0,
        "external_clean_index",
        range(len(kept)),
    )

    kept.to_csv(
        OUT_DIR /
        "external_exact_clean.csv",
        index=False,
    )

    # -------------------------------------------------
    # 4. Summary
    # -------------------------------------------------
    summary = {
        "raw_external_rows":
            int(len(ext)),

        "raw_label_0":
            int((ext["label"] == 0).sum()),

        "raw_label_1":
            int((ext["label"] == 1).sum()),

        "cross_label_conflict_hashes":
            int(len(conflict_hashes)),

        "cross_label_conflict_rows":
            int(len(conflict_rows)),

        "after_conflict_removal_rows":
            int(
                len(ext) - len(conflict_rows)
            ),

        "removed_internal_duplicate_copies":
            int(removed_internal_n),

        "after_internal_dedup_rows":
            int(len(clean)),

        "removed_primary_exact_overlap_rows":
            int(len(removed_overlap)),

        "removed_primary_exact_overlap_label_0":
            int(
                (removed_overlap["label"] == 0)
                .sum()
            ),

        "removed_primary_exact_overlap_label_1":
            int(
                (removed_overlap["label"] == 1)
                .sum()
            ),

        "exact_clean_rows":
            int(len(kept)),

        "exact_clean_label_0":
            int((kept["label"] == 0).sum()),

        "exact_clean_label_1":
            int((kept["label"] == 1).sum()),
    }

    (OUT_DIR / "exact_clean_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n=== EXACT CLEAN SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    for p in sorted(OUT_DIR.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
