from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd
from typing import Iterable

TEXT_EXTS = {".ps1", ".psm1", ".psd1", ".txt", ""}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def read_bytes(path: Path, max_bytes: int = 500_000) -> bytes:
    with path.open("rb") as f:
        return f.read(max_bytes + 1)[:max_bytes]

def decode_script(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "cp1252", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")

def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_EXTS and ".git" not in p.parts:
            yield p

def collect(root: Path, label: int, source: str, subset: str, max_bytes=500_000) -> pd.DataFrame:
    rows = []
    for p in iter_files(root):
        try:
            b = read_bytes(p, max_bytes=max_bytes)
        except OSError:
            continue
        if not b:
            continue
        rows.append({
            "path": str(p),
            "sha256": sha256_bytes(b),
            "label": int(label),
            "source": source,
            "subset": subset,
            "n_bytes": len(b),
        })
    return pd.DataFrame(rows)

def load_text(path: str, max_bytes: int = 500_000) -> str:
    b = read_bytes(Path(path), max_bytes=max_bytes)
    text = decode_script(b)
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
