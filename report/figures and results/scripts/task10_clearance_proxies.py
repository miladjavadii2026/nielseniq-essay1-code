"""
task10_clearance_proxies.py
============================
PURPOSE
-------
Test whether deep-discount (D7) observations near a UPC's final observed weeks
are driven by clearance pricing rather than genuine promotional depth, and
quantify how much of the D7 coefficient is attributable to potential clearance.

BACKGROUND
----------
IS v4 Section 10 (Limitations) flags that deep-bin (D7: 50%+ off) promotions
may include clearance events — a store liquidating a soon-to-be-discontinued
UPC at a heavy discount. Clearance events look like deep promotions but generate
a different behavioral response: (a) consumers may not stock up because the
product won't be available; (b) a post-promotion dip is impossible because the
product exits; (c) the "promotion" is a one-way price reduction, not a temporary
sale cycle.

If clearance is concentrated in D7, the D7 coefficient is contaminated and
the pooled estimate overstates the true deep-discount lift.

CLEARANCE PROXIES (from IS v4, para 255)
------------------------------------------
1. FINAL WEEKS: observations within N weeks of the UPC's last observed week_end
2. STORE COUNT DECLINE: UPC's active store count falling below trailing median
3. DISTRIBUTION BREADTH: distribution points dropping vs prior 8-week average
4. BELOW-MEDIAN PRIOR SALES: trailing 8-week units below the UPC median
5. D7 CLUSTERING: D7 observations overrepresented in final-weeks cluster

DESIGN
------
For each category:
a. Compute last_observed_week per UPC.
b. Create binary clearance flags using each proxy.
c. Estimate FE model separately for:
     - Observations NOT flagged (clean)
     - Observations flagged as potential clearance (dirty)
d. Compare D7 coefficient: if flagged D7 >> clean D7, clearance bias is real.
e. Estimate the "clearance-adjusted" D7 by dropping flagged observations.
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyfixest as pf

# ── Config ─────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task10_clearance"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES    = ["cereal", "coffee", "snacks"]
SAMPLE_FRAC   = 0.04
FINAL_WEEK_N  = 8     # weeks from end to flag as potential clearance
TRAILING_WNDW = 8     # weeks for trailing average (distribution/sales)
RANDOM_SEED   = 42
rng = np.random.default_rng(RANDOM_SEED)

# ── Helper: load sample ───────────────────────────────────────────────────

def load_sampled(cat: str, frac: float) -> pd.DataFrame:
    fpath = os.path.join(STAGE2, f"rms_{cat}_analysis.parquet")
    pf_file = pq.ParquetFile(fpath)
    stores = set()
    for batch in pf_file.iter_batches(100_000, columns=["store_code_uc"]):
        stores.update(batch.column("store_code_uc").to_pylist())
    sampled = set(rng.choice(sorted(stores), max(1, int(len(stores)*frac)), replace=False))
    batches = []
    for batch in pf_file.iter_batches(200_000):
        pdf = batch.to_pandas()
        batches.append(pdf[pdf["store_code_uc"].isin(sampled)])
    return pd.concat(batches, ignore_index=True)


# ── Main ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("Task 10: Clearance Proxy Test")
print("=" * 65)

bins = ["D2", "D3", "D4", "D5", "D6", "D7"]
all_results = []

for cat in CATEGORIES:
    print(f"\n{cat.upper()}")
    panel = load_sampled(cat, SAMPLE_FRAC)
    panel["week_dt"] = pd.to_datetime(panel["week_end"])
    panel["year_week"] = panel["week_dt"].dt.year * 53 + panel["week_dt"].dt.isocalendar().week.astype(int)

    # Depth bin dummies
    for b in bins:
        panel[b] = (panel["depth_bin"] == b).astype(np.int8)

    # ── Proxy 1: proximity to last observed week ─────────────────────────────
    # For each UPC, find its last observed week across all stores in sample
    last_yw = panel.groupby("upc")["year_week"].max().rename("last_yw_upc")
    panel   = panel.merge(last_yw, on="upc", how="left")
    panel["weeks_to_exit"] = panel["last_yw_upc"] - panel["year_week"]
    panel["flag_final"]    = (panel["weeks_to_exit"] <= FINAL_WEEK_N).astype(int)

    # ── Proxy 2: below-median trailing sales ─────────────────────────────────
    panel = panel.sort_values(["store_code_uc", "upc", "year_week"])
    panel["trailing_units"] = (panel
        .groupby(["store_code_uc", "upc"])["units"]
        .transform(lambda x: x.shift(1).rolling(TRAILING_WNDW, min_periods=2).mean()))
    median_by_upc = panel.groupby("upc")["units"].median().rename("median_units_upc")
    panel = panel.merge(median_by_upc, on="upc", how="left")
    panel["flag_low_sales"] = (
        panel["trailing_units"].notna() &
        (panel["trailing_units"] < panel["median_units_upc"] * 0.5)
    ).astype(int)

    # ── Proxy 3: D7 in final weeks (both flags active) ───────────────────────
    panel["flag_clearance"] = (
        (panel["D7"] == 1) &
        (panel["flag_final"] == 1)
    ).astype(int)

    # ── Composite flag: any proxy ─────────────────────────────────────────────
    panel["flag_any"] = ((panel["flag_final"] == 1) |
                         (panel["flag_low_sales"] == 1)).astype(int)

    # Report flag prevalence
    n_d7       = panel["D7"].sum()
    n_d7_final = (panel["D7"] & panel["flag_final"]).sum()
    print(f"  D7 obs total:            {n_d7:,}")
    print(f"  D7 obs in final {FINAL_WEEK_N} wks:  {n_d7_final:,}  ({n_d7_final/max(n_d7,1)*100:.1f}%)")
    print(f"  flag_low_sales:          {panel['flag_low_sales'].sum():,}")
    print(f"  flag_any:                {panel['flag_any'].sum():,}")

    # ── FE estimates: clean vs. flagged ──────────────────────────────────────
    panel["store_upc"]  = panel["store_code_uc"].astype(str) + "_" + panel["upc"].astype(str)
    panel["store_week"] = panel["store_code_uc"].astype(str) + "_" + panel["week_end"].astype(str)
    panel["upc_week"]   = panel["upc"].astype(str) + "_" + panel["week_end"].astype(str)

    if "log_units" not in panel.columns:
        panel["log_units"] = np.log1p(panel["units"])

    rhs = " + ".join(bins)
    formula = f"log_units ~ {rhs} | store_upc + store_week + upc_week"

    for label, mask in [
        ("full",         np.ones(len(panel), dtype=bool)),
        ("clean",        panel["flag_any"] == 0),
        ("flagged",      panel["flag_any"] == 1),
        ("no_clearance", ~((panel["D7"] == 1) & (panel["flag_final"] == 1))),
    ]:
        sub = panel[mask].copy()
        if sub["D7"].sum() < 100:
            print(f"  {label}: too few D7 obs, skip")
            continue
        try:
            fit = pf.feols(formula, data=sub, vcov={"CRV1": "store_code_uc"})
            coefs = dict(fit.coef())
            for b in bins:
                c = coefs.get(b, np.nan)
                s = dict(fit.se()).get(b, np.nan)
                all_results.append({
                    "category": cat, "sample": label, "bin": b,
                    "coef": round(c, 4), "se": round(s, 4), "n_obs": len(sub),
                    "n_d7_obs": int(sub["D7"].sum()),
                })
            print(f"  {label:15s}: D7 = {coefs.get('D7', np.nan):.4f}  [{len(sub):,} obs]")
        except Exception as e:
            print(f"  ERROR ({label}): {e}")

result_df = pd.DataFrame(all_results)
out_path  = os.path.join(OUT_DIR, "task10_clearance_proxy_results.csv")
result_df.to_csv(out_path, index=False)

# Pivot summary for easy reading
if not result_df.empty:
    pivot = (result_df[result_df["bin"] == "D7"]
             .pivot_table(index="category", columns="sample", values="coef")
             .round(4))
    print("\nD7 coefficient by sample (full vs. clean vs. clearance-excluded):")
    print(pivot.to_string())

print(f"\nResults saved: {out_path}")
print("\nTask 10 complete.")
