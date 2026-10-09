"""
task03_bin_monotonicity_tests.py
=================================
PURPOSE
-------
Formally test whether promotional lift is monotonically increasing in discount
depth and whether the depth-response function is concave (diminishing returns).

TESTS PERFORMED
---------------
1. ADJACENT-BIN DIFFERENCES: beta_{j+1} - beta_j > 0 for j = 2..6
   (D2 < D3 < D4 < D5 < D6 < D7 in the cereal baseline)
   t-test using the delta method: t = (b_{j+1} - b_j) / sqrt(se_j^2 + se_{j+1}^2)
   (conservative, ignores covariance — off-diagonal not available in stage4 CSV)

2. STRICT MONOTONICITY CHI-SQUARED: tests that all adjacent differences are > 0
   simultaneously (Wald test with positive-restriction).
   Approximated by counting violations and running a sign test.

3. CURVATURE (SECOND DIFFERENCES): delta^2_j = beta_{j+1} - 2*beta_j + beta_{j-1}
   Negative second difference = concave (diminishing returns).
   Report sign of second difference for each interior bin.

4. LOG-LINEAR DEPTH REGRESSION: regress beta_j on log(midpoint_j)
   R-squared measures how well a log-linear dose-response fits the bin pattern.

INPUT
-----
Stage 4 seven-bin FE tables (already computed — no data reload needed):
  F:/.../stage4_fe_models/tables/fe_seven_bin_{category}.csv

OUTPUT
------
  task03_monotonicity_tests.csv
  task03_curvature.csv
  task03_report.txt
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
STAGE4  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage4_fe_models\tables"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task03_bin_tests"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["cereal", "coffee", "snacks"]

# Bin midpoints (%) for log-linear regression
BIN_MID = {
    "D2": 7.5,   # 5–10%
    "D3": 15.0,  # 10–20%
    "D4": 25.0,  # 20–30%
    "D5": 35.0,  # 30–40%
    "D6": 45.0,  # 40–50%
    "D7": 62.5,  # 50%+ (open-ended; use 62.5 as representative midpoint)
}
BIN_ORDER = ["D2", "D3", "D4", "D5", "D6", "D7"]

print("=" * 65)
print("Task 03: Adjacent-Bin, Monotonicity, and Curvature Tests")
print("=" * 65)

mono_rows = []
curv_rows = []
report_lines = []

for cat in CATEGORIES:
    fpath = os.path.join(STAGE4, f"fe_seven_bin_{cat}.csv")
    if not os.path.exists(fpath):
        print(f"  WARNING: not found — {fpath}")
        continue

    df = pd.read_csv(fpath)
    # Keep primary model and bins D2-D7
    df = df[df["model"] == "seven_bin_primary"].copy()

    # Build a dict: bin -> (estimate, se)
    coefs = {}
    for _, row in df.iterrows():
        t = row["term"].strip()
        if t in BIN_ORDER:
            coefs[t] = (float(row["estimate"]), float(row["std_error"]))

    if not coefs:
        print(f"  WARNING: no bins found in {fpath}")
        continue

    print(f"\n{cat.upper()}")
    print(f"  {'Bin':<5} {'Coef':>8} {'SE':>7}")
    for b in BIN_ORDER:
        c, s = coefs.get(b, (np.nan, np.nan))
        print(f"  {b:<5} {c:>8.4f} {s:>7.4f}")

    # ── 1. Adjacent-bin differences ──────────────────────────────────────
    print(f"\n  Adjacent-bin differences (H0: beta_j+1 = beta_j):")
    for i in range(len(BIN_ORDER) - 1):
        b_lo, b_hi = BIN_ORDER[i], BIN_ORDER[i + 1]
        c_lo, s_lo = coefs.get(b_lo, (np.nan, np.nan))
        c_hi, s_hi = coefs.get(b_hi, (np.nan, np.nan))
        diff = c_hi - c_lo
        se_diff = np.sqrt(s_lo ** 2 + s_hi ** 2)   # conservative (ignores cov)
        t_stat  = diff / se_diff
        p_val   = 2 * stats.norm.sf(abs(t_stat))     # two-sided
        stars   = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        mono_rows.append({
            "category": cat, "bin_lo": b_lo, "bin_hi": b_hi,
            "coef_lo": round(c_lo, 4), "coef_hi": round(c_hi, 4),
            "diff": round(diff, 4), "se_diff_conservative": round(se_diff, 4),
            "t_stat": round(t_stat, 3), "p_value": round(p_val, 4), "sig": stars,
            "monotone": diff > 0,
        })
        direction = "INCREASE" if diff > 0 else "DECREASE"
        print(f"  {b_lo}→{b_hi}: diff={diff:+.4f}  t={t_stat:+.2f}  p={p_val:.4f}{stars}  [{direction}]")

    # Count violations
    n_violations = sum(1 for r in mono_rows if r["category"] == cat and not r["monotone"])
    print(f"  Monotonicity violations: {n_violations}/5")

    # Sign test: under H0 (no pattern), violations should be ~Binomial(5, 0.5)
    n_correct = 5 - n_violations
    p_sign = stats.binomtest(n_correct, 5, 0.5, alternative="greater").pvalue if n_correct > 0 else 1.0
    print(f"  Sign test p-value (H0: random direction): {p_sign:.4f}")

    # ── 2. Curvature (second differences) ────────────────────────────────
    print(f"\n  Curvature (second differences, negative = concave):")
    for i in range(1, len(BIN_ORDER) - 1):
        b_l = BIN_ORDER[i - 1]
        b_c = BIN_ORDER[i]
        b_r = BIN_ORDER[i + 1]
        c_l = coefs.get(b_l, (np.nan,))[0]
        c_c = coefs.get(b_c, (np.nan,))[0]
        c_r = coefs.get(b_r, (np.nan,))[0]
        second_diff = c_r - 2 * c_c + c_l
        shape = "concave" if second_diff < 0 else "convex"
        curv_rows.append({
            "category": cat, "bin_center": b_c,
            "second_diff": round(second_diff, 5), "shape": shape,
        })
        print(f"  {b_c}: Δ²={second_diff:+.5f}  [{shape}]")

    # ── 3. Log-linear fit ─────────────────────────────────────────────────
    log_mids = np.log([BIN_MID[b] for b in BIN_ORDER])
    beta_vals = np.array([coefs.get(b, (np.nan,))[0] for b in BIN_ORDER])
    valid = ~np.isnan(beta_vals)
    if valid.sum() >= 3:
        slope, intercept, r, p, se_slope = stats.linregress(log_mids[valid], beta_vals[valid])
        print(f"\n  Log-linear fit: beta = {intercept:.4f} + {slope:.4f}*log(disc%)")
        print(f"  R-squared = {r**2:.4f}  (how well log-linear describes depth response)")
        report_lines.append(
            f"{cat}: log-linear R^2 = {r**2:.3f}, slope = {slope:.3f}, "
            f"monotone violations = {n_violations}/5, sign-test p = {p_sign:.4f}"
        )

# ── Save outputs ──────────────────────────────────────────────────────────
mono_df = pd.DataFrame(mono_rows)
curv_df = pd.DataFrame(curv_rows)

mono_path = os.path.join(OUT_DIR, "task03_adjacent_bin_tests.csv")
curv_path = os.path.join(OUT_DIR, "task03_curvature.csv")
mono_df.to_csv(mono_path, index=False)
curv_df.to_csv(curv_path, index=False)

txt_path = os.path.join(OUT_DIR, "task03_report.txt")
with open(txt_path, "w") as f:
    f.write("Task 03: Bin Monotonicity and Curvature Summary\n")
    f.write("=" * 50 + "\n")
    f.write("\n".join(report_lines) + "\n\n")
    f.write("Adjacent-bin tests:\n")
    f.write(mono_df.to_string(index=False))
    f.write("\n\nCurvature:\n")
    f.write(curv_df.to_string(index=False))

print(f"\nOutputs:")
print(f"  {mono_path}")
print(f"  {curv_path}")
print(f"  {txt_path}")
print("\nTask 03 complete.")
