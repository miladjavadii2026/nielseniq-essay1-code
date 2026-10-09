#!/usr/bin/env python3
"""Build registry.txt (one governing number per line, at report precision) straight from pipeline
CSVs so nothing is ever retyped. verify_report.py --registry then demands every line verbatim in
the rendered PDF. Usage:
  build_registry.py --fe fe_seven_bin_cereal.csv [--fe ...] [--offsets offset_table.csv]
                    [--dml results.csv] [--extra numbers.txt] --out registry.txt
FE files: columns term,estimate,std_error (4 dp). Offsets: output of offset_table.py (3 dp abs
ratios). DML: any CSV with 'category' plus theta|ate|estimate (3 dp) and optional se columns."""
import csv, sys
args = sys.argv[1:]; get = lambda k: [args[i + 1] for i, a in enumerate(args) if a == k]
out, lines = get("--out")[0], []
for f in get("--fe"):
    for r in csv.DictReader(open(f)):
        lines += [f"{float(r['estimate']):.4f}", f"{float(r['std_error']):.4f}"]
for f in get("--offsets"):
    for r in csv.DictReader(open(f)): lines.append(f"{float(r['abs_ratio']):.3f}")
for f in get("--dml"):
    for r in csv.DictReader(open(f)):
        v = next((r[k] for k in ("theta", "ate", "estimate") if k in r and r[k]), None)
        if v: lines.append(f"{float(v):.3f}")
        s = next((r[k] for k in ("se_cluster", "se", "std_error") if k in r and r[k]), None)
        if s: lines.append(f"{float(s):.3f}")
for f in get("--extra"): lines += [l.strip() for l in open(f) if l.strip()]
lines = list(dict.fromkeys(lines))                      # dedupe, keep order
open(out, "w").write("\n".join(lines) + "\n")
print(f"registry: {len(lines)} numbers -> {out}")
