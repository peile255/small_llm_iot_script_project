from pathlib import Path
import argparse, json

def pct(v):
    return r"\textit{TBD}" if v is None else f"{100*v:.2f}\\%"

def num(v, digits=3):
    return r"\textit{TBD}" if v is None else f"{v:.{digits}f}"

def get(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.results).read_text())

    b = s.get("baseline", {})
    q = s.get("qwen2_7b", {})
    m = s.get("mistral_7b", {})
    p = s.get("phi3_mini", {})

    lines = [r"% AUTO-GENERATED. DO NOT EDIT BY HAND.", r"\newcommand{\NA}{\textit{TBD}}"]

    def cmd(name, value):
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    for prefix, obj, is_llm in [
        ("Baseline", b, False),
        ("Qwen", q, True),
        ("Mistral", m, True),
        ("Phi", p, True),
    ]:
        basepath = ("test",) if is_llm else ()
        cmd(prefix+"Accuracy", pct(get(obj, *basepath, "accuracy")))
        cmd(prefix+"Precision", pct(get(obj, *basepath, "precision")))
        cmd(prefix+"Recall", pct(get(obj, *basepath, "recall")))
        cmd(prefix+"Fone", pct(get(obj, *basepath, "f1")))
        cmd(prefix+"FPR", pct(get(obj, *basepath, "fpr")))
        cmd(prefix+"MCC", num(get(obj, *basepath, "mcc")))

    for prefix, obj in [("Qwen",q),("Mistral",m),("Phi",p)]:
        cmd(prefix+"MixedRecall", pct(get(obj,"mixed","recall")))
        cmd(prefix+"MixedFNR", pct(get(obj,"mixed","fnr")))
        cmd(prefix+"ExternalRecall", pct(get(obj,"external","recall")))
        cmd(prefix+"Invalid", pct(get(obj,"test","invalid_output_rate")))
        cmd(prefix+"Latency", num(get(obj,"test","median_latency_s"),3))
        cmd(prefix+"Throughput", num(get(obj,"test","throughput_files_s"),3))
        cmd(prefix+"VRAM", num(get(obj,"peak_vram_gb"),2))

    Path(args.out).write_text("\n".join(lines)+"\n")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
