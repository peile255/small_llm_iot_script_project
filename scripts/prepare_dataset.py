from pathlib import Path
import argparse, sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import collect

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Reject conflicting exact hashes rather than silently choosing a label.
    conflicts = df.groupby("sha256")["label"].nunique()
    bad = set(conflicts[conflicts > 1].index)
    if bad:
        print(f"[warning] dropping {len(bad)} exact hashes with conflicting labels")
        df = df[~df.sha256.isin(bad)]
    return df.drop_duplicates("sha256").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpsd", required=True)
    ap.add_argument("--external", required=False)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mpsd = Path(args.mpsd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    benign = collect(mpsd / "powershell_benign_dataset", 0, "mpsd", "benign")
    mal = collect(mpsd / "malicious_pure", 1, "mpsd", "malicious_pure")
    mixed = collect(mpsd / "mixed_malicious", 1, "mpsd", "mixed_malicious")

    primary = dedup(pd.concat([benign, mal], ignore_index=True))
    mixed = dedup(mixed)
    primary_hashes = set(primary.sha256) | set(mixed.sha256)

    if len(primary) == 0:
        raise RuntimeError("No MPSD samples found. Check repository paths.")

    train_val, test = train_test_split(
        primary, test_size=0.20, random_state=args.seed, stratify=primary.label
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=args.seed, stratify=train_val.label
    )  # 0.125 of 80% = 10%

    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    mixed.to_csv(out / "mixed_test.csv", index=False)

    if args.external:
        ext_root = Path(args.external)
        candidates = []
        for dirname in ["malicious_samples", "bazaar_out", "hybrid_analysis_out", "triage_out"]:
            p = ext_root / dirname
            if p.exists():
                candidates.append(collect(p, 1, "fa2y_external", dirname))
        ext = dedup(pd.concat(candidates, ignore_index=True)) if candidates else pd.DataFrame()
        if not ext.empty:
            before = len(ext)
            ext = ext[~ext.sha256.isin(primary_hashes)].copy()
            print(f"External exact-overlap removal: {before-len(ext)} files removed.")
        ext.to_csv(out / "external_malicious.csv", index=False)
    else:
        pd.DataFrame(columns=primary.columns).to_csv(out / "external_malicious.csv", index=False)

    manifest = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
        mixed.assign(split="mixed_test"),
    ], ignore_index=True)
    manifest.to_csv(out / "manifest.csv", index=False)

    print({
        "train": len(train), "val": len(val), "test": len(test), "mixed": len(mixed),
        "class_counts": primary.label.value_counts().to_dict()
    })

if __name__ == "__main__":
    main()
