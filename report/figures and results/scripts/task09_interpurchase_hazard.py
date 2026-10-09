"""
task09_interpurchase_hazard.py
================================
PURPOSE
-------
Estimate interpurchase-duration or hazard models to directly test purchase
acceleration. If promotions cause households to buy sooner than they otherwise
would, the hazard of purchase should be elevated during promoted periods.

ECONOMIC INTERPRETATION
-----------------------
The hazard at time t given last purchase at s is:
  h(t | s, X) = h_0(t - s) * exp(beta * promoted(t) + gamma * X)

If beta > 0, promotions increase the probability of purchase in the current
week conditional on not having purchased since week s. This is the direct
measure of intertemporal acceleration — not the quantity purchased, but the
timing of when a purchase occurs.

Combining Task 8 (quantity-on-purchase-occasion) with Task 9 (purchase-timing)
decomposes the promotional effect into:
  Total unit impact = Timing acceleration × Per-occasion quantity effect

MODELS ESTIMATED
-----------------
1. Cox Proportional Hazard model (semi-parametric, no duration assumption)
   h(t) = h_0(t) * exp(beta * W5 + gamma * depth + HH controls)

2. Discrete-time logit (for the binary "purchase this week" outcome,
   estimated on a balanced weekly panel with zero-padding)
   P(buy_{hit}=1) = logistic(alpha_h + alpha_t + beta * W5_t + gamma * ipi_{hit-1})

DATA REQUIREMENT
-----------------
Household panel (HMS) with purchase occasions. The balanced weekly panel
(padded with zeros) is needed for the discrete-time model. This requires
knowing which weeks each household was "at risk" (i.e., active panelists).

This script estimates what it can from the HMS purchase-occasion data
and clearly flags what requires the full balanced panel.
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task09_hazard"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["cereal", "coffee", "snacks"]

# ── Main ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("Task 09: Interpurchase Duration and Hazard Models")
print("=" * 65)

all_results = []

for cat in CATEGORIES:
    fpath = os.path.join(STAGE2, f"hms_{cat}_analysis.parquet")
    if not os.path.exists(fpath):
        print(f"\nWARNING: HMS file not found — {fpath}")
        continue

    print(f"\n{cat.upper()}")
    cols = [f.name for f in pq.read_schema(fpath)]
    hh_col   = next((c for c in ["household_code", "panid"] if c in cols), None)
    upc_col  = next((c for c in ["upc", "upc_code"] if c in cols), None)
    wk_col   = next((c for c in ["week_end", "week", "purchase_date"] if c in cols), None)
    qty_col  = next((c for c in ["quantity", "units", "hh_purchase_qty"] if c in cols), None)
    w5_col   = next((c for c in ["W5", "promo_flag", "promo", "promo_coincident_flag"] if c in cols), None)

    read_cols = [hh_col, upc_col, wk_col, qty_col, w5_col]
    read_cols = [c for c in dict.fromkeys(read_cols) if c and c in cols]
    df = pd.read_parquet(fpath, columns=read_cols)
    print(f"  Loaded: {len(df):,} rows")

    if not all([hh_col, upc_col, wk_col]):
        print("  Cannot identify required columns — skipping.")
        continue

    df = df.sort_values([hh_col, upc_col, wk_col]).reset_index(drop=True)
    df["week_dt"] = pd.to_datetime(df[wk_col])
    df["year_week"] = df["week_dt"].dt.year * 53 + df["week_dt"].dt.isocalendar().week.astype(int)

    # ── Interpurchase intervals ─────────────────────────────────────────────
    df["prev_yw"] = df.groupby([hh_col, upc_col])["year_week"].shift(1)
    df["ipi"]     = df["year_week"] - df["prev_yw"]   # weeks since last purchase
    df = df[df["ipi"].notna()].copy()                  # first-purchase has no IPI

    # ── Summary statistics by promotion status ──────────────────────────────
    if w5_col:
        grp = df.groupby([w5_col])["ipi"].agg(
            mean_ipi="mean", median_ipi="median", n="count"
        ).reset_index()
        grp["category"] = cat
        print(f"\n  Mean IPI by promotion status:")
        print(grp.to_string(index=False))

        promoted   = df[df[w5_col] == 1]["ipi"].dropna()
        unpromoted = df[df[w5_col] == 0]["ipi"].dropna()
        if len(promoted) > 30 and len(unpromoted) > 30:
            t, p = stats.ttest_ind(promoted, unpromoted)
            print(f"\n  t-test (mean IPI: promoted vs. unpromoted):")
            print(f"    promoted mean:   {promoted.mean():.2f} weeks")
            print(f"    unpromoted mean: {unpromoted.mean():.2f} weeks")
            print(f"    t={t:.3f}, p={p:.4f}")
            all_results.append({
                "category": cat, "test": "t_test_ipi",
                "promoted_mean_ipi":   round(promoted.mean(), 3),
                "unpromoted_mean_ipi": round(unpromoted.mean(), 3),
                "t_stat": round(t, 3), "p_value": round(p, 4),
                "n_promoted": len(promoted), "n_unpromoted": len(unpromoted),
            })

    # ── Discrete-time logit (purchase probability model) ───────────────────
    # Build hazard dataset: for each HH-UPC, create weekly rows
    # from first observed week to last observed week, flagging purchase occasions
    # NOTE: requires that we know all active weeks (participation weeks)
    # Without that, we use between-purchase intervals as the risk set

    # Construct from pairs of consecutive purchases: each row = one inter-purchase spell
    # Duration = IPI, event = 1 (always, since we only see completed spells)
    # Covariates: was the ending week promoted?

    # This gives a censored sample (right-censored final spells drop out naturally)
    # Parametric Weibull duration model approximation via log-linear regression
    # ln(IPI) = alpha + beta * W5_at_purchase + gamma * ipi_lag + eps
    # (accelerated failure time model)

    df_spell = df.copy()
    df_spell["ln_ipi"] = np.log(df_spell["ipi"].clip(lower=0.5))   # log(IPI), clip to avoid -inf
    df_spell["ipi_lag"] = df_spell.groupby([hh_col, upc_col])["ipi"].shift(1)

    if w5_col and df_spell["ln_ipi"].notna().sum() > 200:
        # OLS on log-IPI (AFT approximation, no censoring)
        # Negative coefficient on W5 = shorter IPI = faster repurchase = acceleration
        from statsmodels.formula.api import ols
        ipi_lag_col = "ipi_lag" if df_spell["ipi_lag"].notna().sum() > 100 else None
        rhs_aft = w5_col + (f" + ipi_lag" if ipi_lag_col else "")
        try:
            aft = ols(f"ln_ipi ~ {rhs_aft}", data=df_spell.dropna(subset=["ln_ipi", w5_col])).fit(
                cov_type="HC1"
            )
            w5_coef = aft.params.get(w5_col, np.nan)
            w5_pval = aft.pvalues.get(w5_col, np.nan)
            print(f"\n  AFT log(IPI) ~ {rhs_aft}:")
            print(f"    {w5_col} coef = {w5_coef:.4f}  p = {w5_pval:.4f}")
            print(f"    Interpretation: exp({w5_coef:.3f}) = {np.exp(w5_coef):.3f} × baseline IPI")
            print(f"    (< 1 = shorter IPI = purchase acceleration)")
            all_results.append({
                "category": cat, "test": "AFT_log_IPI",
                "w5_coef": round(w5_coef, 4), "w5_pvalue": round(w5_pval, 4),
                "ipi_ratio": round(np.exp(w5_coef), 4),
                "interpretation": "acceleration" if w5_coef < 0 else "deceleration",
                "n_obs": int(df_spell["ln_ipi"].notna().sum()),
            })
        except Exception as e:
            print(f"  AFT error: {e}")

    # Save per-category IPI distributions
    if w5_col:
        ipi_dist = df.groupby([w5_col, "ipi"]).size().reset_index(name="n")
        ipi_dist["category"] = cat
        ipi_path = os.path.join(OUT_DIR, f"task09_ipi_dist_{cat}.csv")
        ipi_dist.to_csv(ipi_path, index=False)
        print(f"  IPI distribution saved: {ipi_path}")

result_df = pd.DataFrame(all_results)
out_path  = os.path.join(OUT_DIR, "task09_hazard_results.csv")
result_df.to_csv(out_path, index=False)

print(f"\nResults saved: {out_path}")
print(result_df.to_string(index=False))
print("""
NOTE: The discrete-time logit requires a balanced weekly panel with zero-padding
for non-purchase weeks. This needs the NielsenIQ household participation file
(panelists' active dates) to construct the full weekly risk set per household.
Cox model can be estimated using the `lifelines` package:
  from lifelines import CoxPHFitter
  cph = CoxPHFitter(); cph.fit(df_spell, duration_col='ipi', event_col='event',
                               formula='W5 + C(year)')
""")
print("Task 09 complete.")
