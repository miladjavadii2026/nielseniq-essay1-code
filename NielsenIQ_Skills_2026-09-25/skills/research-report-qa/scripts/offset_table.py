#!/usr/bin/env python3
"""Reviewer-proof offset table from raw event-study coefficients. For each category and horizon H
prints the exact event times, the baseline (mean of t=-8..-5), the denominator sum over t=-4..0
and numerator sum over t=1..H (both baseline-adjusted), and the signed ratio. This is the single
code-generated table the offset claims rest on; never retype these numbers.
Usage: offset_table.py EVENT_COEFS.csv [--horizons 4,8,12] [--out table.csv]
Input columns: category, event_time, coef (FINAL_RERUN_v12/02_event_study/02_event_study_coefs.csv)."""
import csv, sys, collections
src = sys.argv[1]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
H = [int(h) for h in arg("--horizons", "4,8,12").split(",")]
by = collections.defaultdict(dict)
for r in csv.DictReader(open(src)):
    by[r["category"]][int(r["event_time"])] = float(r["coef"])
rows = []
for cat, d in by.items():
    base = sum(d[t] for t in range(-8, -4)) / 4
    den_t = list(range(-4, 1)); den = sum(d[t] - base for t in den_t)
    for h in H:
        num_t = list(range(1, h + 1)); num = sum(d[t] - base for t in num_t)
        ratio = num / den                                   # full precision; round ONCE at print time
        rows.append(dict(category=cat, horizon_weeks=h, baseline=f"{base:.4f}",
                         denominator_times=",".join(f"t{t:+d}" for t in den_t), denominator_sum=f"{den:.4f}",
                         numerator_times=",".join(f"t{t:+d}" for t in num_t), numerator_sum=f"{num:.4f}",
                         signed_ratio=f"{ratio:.3f}", abs_ratio=f"{abs(ratio):.3f}", raw_ratio=repr(ratio)))
for r in rows:
    print(f"{r['category']:8s} {r['horizon_weeks']:>2d}wk base={r['baseline']} den={r['denominator_sum']} "
          f"num={r['numerator_sum']} ratio={r['signed_ratio']} (raw {r['raw_ratio'][:9]})")
if arg("--out", None):
    with open(arg("--out", None), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", arg("--out", None))
