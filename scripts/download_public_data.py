from pathlib import Path
import argparse
import subprocess

REPOS = {
    "mpsd": "https://github.com/das-lab/mpsd.git",
    "Malicious-PowerShell-Dataset": "https://github.com/Fa2y/Malicious-PowerShell-Dataset.git",
}

def clone(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.root)
    for name, url in REPOS.items():
        clone(url, root / name)
    print("Downloaded public repositories. DO NOT execute any sample files.")

if __name__ == "__main__":
    main()
