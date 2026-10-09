#!/usr/bin/env python3
"""Stage 7, Step 2: inventory every results record so Appendix B and the
results-records package are written from a file list, not from memory.

Usage:
  python results_inventory.py --base "E:\\NielsenIQ_Phase2" --out DIR [--hash]

Walks 06_results (plus FINAL_RERUN_v12 and 01_documentation audits), tags each
file with stage folder, category, kind (table, figure, model, marker, log, doc)
and writes appendix_b_inventory.csv and .md. --hash adds sha256 for files under
50 MB so a later run can prove an exhibit's source file did not change.
Never writes under G:.
"""
from __future__ import annotations
import argparse, csv, hashlib, sys
from datetime import datetime
from pathlib import Path

CATS = ("cereal", "snacks", "coffee")
KIND = {".csv": "table", ".parquet": "table", ".png": "figure", ".pdf": "figure", ".pkl": "model",
        ".json": "audit", ".txt": "log", ".log": "log", ".md": "doc", ".docx": "doc", ".html": "doc"}
ROOTS = ["06_results", "FINAL_RERUN_v12", "01_documentation"]
SKIP = {"_tmp_years", "__pycache__", "FINAL_RERUN_v12_C_outputs"}


def on_g(p: Path) -> bool:
    s = str(p).replace("/", "\\").upper()
    return s.startswith("G:") or "\\G\\NIELSENIQ" in s or s.startswith("\\MNT\\USER-DATA\\UPLOADS\\G")


def sha(p: Path, limit=50 * 2**20):
    if p.stat().st_size > limit:
        return "skipped (>50MB)"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--hash", action="store_true")
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    if on_g(out):
        sys.exit("Refusing to write under G:.")
    rows = []
    for root in ROOTS:
        rdir = base / root
        if not rdir.exists():
            continue
        for f in sorted(rdir.rglob("*")):
            if not f.is_file() or any(part in SKIP for part in f.relative_to(base).parts):
                continue
            rel = f.relative_to(base)
            low = str(rel).lower()
            if root == "01_documentation" and "audit" not in low and "diagnostic" not in low:
                continue
            stage = rel.parts[1] if root == "06_results" and len(rel.parts) > 2 else (root + " (root)" if root == "06_results" else root)
            cat = next((c for c in CATS if c in low), "")
            kind = "marker" if f.name.startswith(".") else KIND.get(f.suffix.lower(), "other")
            st = f.stat()
            rows.append(dict(path=str(rel).replace("\\", "/"), stage=stage, category=cat, kind=kind,
                             bytes=st.st_size, modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                             sha256_16=sha(f) if a.hash else "", supports_exhibit=""))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "appendix_b_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["# Appendix B. Results records index", "", f"Base: `{base}`; {len(rows)} files; generated {datetime.now().isoformat(timespec='seconds')}", "",
          "| Path | Stage | Category | Kind | Bytes | Modified | Supports exhibit |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['path']} | {r['stage']} | {r['category']} | {r['kind']} | {r['bytes']} | {r['modified']} | {r['supports_exhibit']} |" for r in rows]
    (out / "appendix_b_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    by = {}
    for r in rows:
        by[(r["stage"], r["kind"])] = by.get((r["stage"], r["kind"]), 0) + 1
    for (s, k), n in sorted(by.items()):
        print(f"  {s:<32} {k:<7} {n}")
    print(f"Written: {out / 'appendix_b_inventory.csv'} and .md ({len(rows)} files)")
    print("Fill the supports_exhibit column (Table 3, Figure A2, ...) before it becomes Appendix B.")


if __name__ == "__main__":
    main()
