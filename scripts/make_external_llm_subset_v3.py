#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import tarfile

import pandas as pd


SEED = 20260922
N_PER_CLASS = 1000

ROOT = Path(
    "outputs/paper_v3/frozen_external_v1"
)

FINAL = ROOT / "external_final_audited.csv"

ARCHIVE = Path(
    "data/external/wwe123/ast_ps1_by_label.tar.gz"
)

OUT_DIR = Path(
    "data/processed/external_v3"
)

FILES_DIR = OUT_DIR / "llm_subset_files"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FILES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():
    df = pd.read_csv(FINAL)

    assert len(df) == 19632
    assert (df["label"] == 0).sum() == 9324
    assert (df["label"] == 1).sum() == 10308

    parts = []

    for label in [0, 1]:
        sub = (
            df[df["label"] == label]
            .sample(
                n=N_PER_CLASS,
                random_state=SEED,
                replace=False,
            )
            .copy()
        )

        parts.append(sub)

    subset = pd.concat(
        parts,
        ignore_index=True,
    )

    # Deterministic final ordering.
    subset = (
        subset
        .sort_values(
            ["label", "archive_path"]
        )
        .reset_index(drop=True)
    )

    subset.insert(
        0,
        "external_llm_index",
        range(len(subset)),
    )

    assert len(subset) == 2000
    assert (subset["label"] == 0).sum() == 1000
    assert (subset["label"] == 1).sum() == 1000

    wanted = set(
        subset["archive_path"]
        .astype(str)
    )

    extracted = {}

    print(
        "Extracting selected scripts...",
        flush=True,
    )

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

            row = subset[
                subset["archive_path"]
                == member.name
            ].iloc[0]

            label = int(row["label"])

            # Use SHA as filename to avoid collisions
            # and preserve deterministic provenance.
            sha = str(row["raw_sha256"])

            label_dir = FILES_DIR / str(label)
            label_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            path = label_dir / f"{sha}.ps1"

            path.write_bytes(raw)

            extracted[
                member.name
            ] = str(path)

    missing = wanted - set(extracted)

    if missing:
        raise RuntimeError(
            f"Missing extracted files: {len(missing)}"
        )

    subset["path"] = (
        subset["archive_path"]
        .map(extracted)
    )

    assert subset["path"].notna().all()

    manifest = (
        OUT_DIR /
        "external_llm_subset_2000.csv"
    )

    subset.to_csv(
        manifest,
        index=False,
    )

    summary = {
        "seed": SEED,
        "n_total": int(len(subset)),
        "n_label_0":
            int((subset["label"] == 0).sum()),
        "n_label_1":
            int((subset["label"] == 1).sum()),
        "source_final_audited_rows":
            int(len(df)),
        "selection":
            "stratified random sample without replacement",
    }

    (
        OUT_DIR /
        "external_llm_subset_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    # Freeze hashes
    hashes = []

    for p in sorted(
        FILES_DIR.rglob("*.ps1")
    ):
        hashes.append(
            f"{sha256_file(p)}  {p}\n"
        )

    (
        OUT_DIR /
        "external_llm_subset_SHA256SUMS.txt"
    ).write_text(
        "".join(hashes)
    )

    print("\n=== EXTERNAL LLM SUBSET ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(manifest)
    print(
        OUT_DIR /
        "external_llm_subset_summary.json"
    )
    print(
        OUT_DIR /
        "external_llm_subset_SHA256SUMS.txt"
    )


if __name__ == "__main__":
    main()
