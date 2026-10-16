from pathlib import Path
import argparse, json

def load_json(p):
    return json.loads(Path(p).read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default="outputs/summary.json")
    args = ap.parse_args()
    root = Path(args.outputs)
    summary = {}

    b = root / "baseline" / "metrics.json"
    if b.exists():
        summary["baseline"] = load_json(b)

    for key in ["qwen2_7b", "mistral_7b", "phi3_mini"]:
        p = root / key / "metrics.json"
        if p.exists():
            summary[key] = load_json(p)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
