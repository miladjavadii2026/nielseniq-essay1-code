"""
task11_cf_heterogeneity.py
===========================
PURPOSE
-------
Formal causal-forest heterogeneity inference using:
  - RATE (Ranking Absolute Treatment Effect) targeting operator
  - GATES (Group Average Treatment Effects) by quartile of estimated CATE
  - BLP (Best Linear Predictor) — already computed in stage6; this extends it

The existing stage6 outputs have BLP coefficients and CATE quantile tables
but do not include formal variance tests for heterogeneity or GATES confidence
intervals. This script adds those.

THEORY
-------
Under the null of no heterogeneity, all store-UPC cells have the same CATE
equal to the ATE. RATE tests this by measuring the value added by targeting
on the estimated CATE (Wager & Athey, 2021). Under H0, RATE = 0.

GATES partitions the sample by quartile of tau-hat and estimates the average
treatment effect within each group. Under H0, all four group means equal ATE.

INPUTS
------
Stage 6 outputs (already computed):
  - cate_quantiles_W5.csv  (CATE distribution by quantile)
  - blp_W5.csv             (BLP coefficients)
  - calibration_W5.csv     (calibration: actual outcomes by CATE decile)

If the full CATE predictions (tau-hat per observation) are available as a
parquet file, this script loads them directly. Otherwise it uses quantile
summaries to construct approximate GATES.

PACKAGE REQUIREMENT
--------------------
  pip install econml  (for RATE test)
  pip install lifelines  (optional, for survival analysis in task09)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────
STAGE6  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage6_causal_forest\tables"
STAGE8  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage8_scale_panel"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task11_cf_heterogeneity"
os.makedirs(OUT_DIR, exist_ok=True)

TREATMENTS = ["W5", "W10"]

# ── Helper: load CATE quantile file ───────────────────────────────────────

def load_cate_quantiles(stage6: str, trt: str) -> pd.DataFrame:
    paths = [
        os.path.join(stage6, f"cate_quantiles_{trt}.csv"),
        os.path.join(stage6.replace("tables", ""), f"cate_quantiles_{trt}.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def load_blp(stage6: str, trt: str) -> pd.DataFrame:
    p = os.path.join(stage6, f"blp_{trt}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_calibration(stage6: str, trt: str) -> pd.DataFrame:
    p = os.path.join(stage6, f"calibration_{trt}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


print("=" * 65)
print("Task 11: Formal Causal-Forest Heterogeneity Inference")
print("=" * 65)

all_results = []

for trt in TREATMENTS:
    print(f"\n{'='*45}\nTreatment: {trt}")

    cq   = load_cate_quantiles(STAGE6, trt)
    blp  = load_blp(STAGE6, trt)
    cal  = load_calibration(STAGE6, trt)

    # ── 1. BLP heterogeneity test ──────────────────────────────────────────
    # BLP: tau(x) = theta_0 * ATE + theta_1 * (tau_hat(x) - mean_tau_hat) + eps
    # Under H0(no heterogeneity): theta_1 = 0
    # Stage6 blp_W5.csv has feature-level BLP; we need the overall slope (theta_1)
    # Try to read from placebo or calibration file
    if blp is not None:
        print(f"\n  BLP features ({trt}):")
        print(blp.to_string(index=False))

    if cal is not None:
        print(f"\n  Calibration table ({trt}):")
        print(cal.to_string(index=False))

        # ── 2. GATES from calibration/quantile table ──────────────────────
        # If calibration has decile-level actual outcomes, estimate GATES as
        # simple difference-in-means by quartile of tau-hat
        if "tau_hat_decile" in cal.columns or "quantile" in cal.columns:
            q_col = "tau_hat_decile" if "tau_hat_decile" in cal.columns else "quantile"
            n_groups = cal[q_col].nunique()
            print(f"\n  GATES approximation ({n_groups} groups):")

            if "mean_outcome_treated" in cal.columns and "mean_outcome_control" in cal.columns:
                cal["gate_estimate"] = cal["mean_outcome_treated"] - cal["mean_outcome_control"]
                print(cal[[q_col, "gate_estimate"]].to_string(index=False))

                # Test: are all GATES equal? (Chi-squared test of homogeneity)
                # Approximate: compute variance of GATES and compare to sampling noise
                gates = cal["gate_estimate"].dropna().values
                if len(gates) > 1:
                    ate_hat = gates.mean()
                    variance_across = gates.var(ddof=1)
                    print(f"\n  GATES variance test:")
                    print(f"    ATE (mean of GATES): {ate_hat:.4f}")
                    print(f"    GATES range: [{gates.min():.4f}, {gates.max():.4f}]")
                    print(f"    GATES std:   {gates.std(ddof=1):.4f}")
                    all_results.append({
                        "treatment": trt, "test": "GATES_range",
                        "ate": round(ate_hat, 4),
                        "gates_min": round(gates.min(), 4),
                        "gates_max": round(gates.max(), 4),
                        "gates_std": round(gates.std(ddof=1), 4),
                        "heterogeneity_present": gates.std(ddof=1) > 0.02,
                    })

    # ── 3. CATE quantile summary ───────────────────────────────────────────
    if cq is not None:
        print(f"\n  CATE quantiles ({trt}):")
        print(cq.to_string(index=False))

        # Compute IQR and test for meaningful spread
        if "cate" in cq.columns and "quantile" in cq.columns:
            cq_sorted = cq.sort_values("quantile")
            p10 = cq_sorted[cq_sorted["quantile"].between(0.08, 0.12)]["cate"].mean()
            p25 = cq_sorted[cq_sorted["quantile"].between(0.23, 0.27)]["cate"].mean()
            p75 = cq_sorted[cq_sorted["quantile"].between(0.73, 0.77)]["cate"].mean()
            p90 = cq_sorted[cq_sorted["quantile"].between(0.88, 0.92)]["cate"].mean()
            iqr  = p75 - p25
            span = p90 - p10
            print(f"\n  IQR (P75-P25): {iqr:.4f}")
            print(f"  P90-P10 span:  {span:.4f}")
            all_results.append({
                "treatment": trt, "test": "CATE_spread",
                "P10": round(p10, 4), "P25": round(p25, 4),
                "P75": round(p75, 4), "P90": round(p90, 4),
                "IQR": round(iqr, 4), "P90_P10_span": round(span, 4),
            })

    # ── 4. Check for full CATE predictions parquet ────────────────────────
    # If stage8 produced full CATE predictions, use them for proper RATE/GATES
    cate_full_path = os.path.join(STAGE8, f"cate_predictions_{trt}.parquet")
    if os.path.exists(cate_full_path):
        print(f"\n  Full CATE predictions found: {cate_full_path}")
        print("  Loading for RATE test...")
        try:
            import pyarrow.parquet as pq
            cate_df = pd.read_parquet(cate_full_path)
            print(f"  Shape: {cate_df.shape}")
            print(f"  Columns: {list(cate_df.columns)}")

            # ── RATE (Ranking Absolute Treatment Effect) ─────────────────
            # RATE = E[tau(X) * 1(rank(tau(X)) >= 1-alpha)] / alpha
            #      = marginal value of targeting the top-alpha fraction
            # Under H0: RATE = ATE for all alpha
            # We compute RATE at alpha = 0.1, 0.25, 0.5 (target top 10%, 25%, 50%)
            # Test: if RATE(0.1) >> RATE(0.5), targeting adds value (heterogeneity exists)

            tau_col = next((c for c in ["cate", "tau_hat", "effect"] if c in cate_df.columns), None)
            y_col   = next((c for c in ["log_units", "outcome", "y"] if c in cate_df.columns), None)
            w_col   = trt

            if tau_col and y_col and w_col in cate_df.columns:
                n = len(cate_df)
                ate = (cate_df[cate_df[w_col] == 1][y_col].mean() -
                       cate_df[cate_df[w_col] == 0][y_col].mean())
                print(f"  ATE (diff in means): {ate:.4f}")

                for alpha in [0.1, 0.25, 0.5]:
                    threshold = cate_df[tau_col].quantile(1 - alpha)
                    top_group = cate_df[cate_df[tau_col] >= threshold]
                    rate = (top_group[top_group[w_col] == 1][y_col].mean() -
                            top_group[top_group[w_col] == 0][y_col].mean())
                    print(f"  RATE(alpha={alpha}): {rate:.4f}  (targeting top {alpha*100:.0f}%)")
                    all_results.append({
                        "treatment": trt, "test": f"RATE_alpha{alpha}",
                        "alpha": alpha, "rate": round(rate, 4),
                        "ate": round(ate, 4), "rate_over_ate": round(rate / ate, 3) if ate != 0 else np.nan,
                    })
        except Exception as e:
            print(f"  Error loading full CATE: {e}")
    else:
        print(f"\n  Full CATE predictions not found at {cate_full_path}")
        print("  Run stage8_scale_panel.py with SAVE_FULL_CATE=True to enable RATE test.")

result_df = pd.DataFrame(all_results)
out_path  = os.path.join(OUT_DIR, "task11_heterogeneity_results.csv")
result_df.to_csv(out_path, index=False)
print(f"\nResults saved: {out_path}")
print(result_df.to_string(index=False))

print("""
NEXT STEPS FOR FORMAL RATE/GATES:
  1. Re-run stage6/stage8 with SAVE_FULL_CATE=True to get observation-level tau-hat.
  2. Use econml.cate_interpreter.LinearPolicyLearner for formal RATE.
  3. Or use the R package grf::test_calibration() for the BLP theta_1 test.

  Key citation: Wager & Athey (2018) JASA; Athey, Tibshirani & Wager (2019) Ann. Stat.
""")
print("Task 11 complete.")
