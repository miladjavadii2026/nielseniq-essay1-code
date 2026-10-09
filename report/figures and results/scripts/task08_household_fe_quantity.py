"""
task08_household_fe_quantity.py
================================
PURPOSE
-------
Estimate household and household-UPC fixed-effects quantity models to directly
measure stockpiling at the household level.

The store-level (RMS) evidence shows post-promotion dips in aggregate sales.
But this confounds: (1) individual households buying more and then less
(stockpiling), (2) promotion attracting one-time buyers who then disappear.
The household panel (HMS) resolves this by following the same household over
time, so within-household variation identifies stockpiling directly.

ESTIMATING EQUATIONS
---------------------
Model A: Household FE
  ln(qty+1)_{hit} = beta * W5_{it} + alpha_h + alpha_t + epsilon_{hit}

Model B: Household × UPC FE
  ln(qty+1)_{hit} = sum_j [beta_j * D_j_{it}] + alpha_{hu} + alpha_t + epsilon_{hit}

Model C: Interpurchase effect (quantity acceleration)
  ln(qty+1)_{hit} = beta * W5_{it} + gamma * (weeks_since_last_buy_{hit})
                  + alpha_{hu} + alpha_t + epsilon_{hit}
  Tests whether promotional purchases are larger than typical purchases
  controlling for interpurchase interval.

DATA
----
HMS parquet: hms_cereal_analysis.parquet  (household-level panel)
Schema (from stage2 output):
  household_code, upc, week_end, units, unit_price, coupon_value,
  promo_flag (patched), quantity, panel_year, depth_bin, W5, W10, log_units

NOTE: HMS data has sparser coverage (purchase occasions only, not all weeks).
All weeks with no purchase are implicitly missing (not zero-padded).
This means HMS estimates E[qty | purchase occasion] — the purchase-conditional
quantity effect. Household FE absorbs household size, income, etc.
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyfixest as pf

# ── Config ─────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task08_household_fe"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["cereal", "coffee", "snacks"]

# ── Main ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("Task 08: Household FE Quantity Models")
print("=" * 65)

all_results = []

for cat in CATEGORIES:
    fpath = os.path.join(STAGE2, f"hms_{cat}_analysis.parquet")
    if not os.path.exists(fpath):
        print(f"\nWARNING: HMS file not found — {fpath}")
        continue

    print(f"\n{cat.upper()}")

    # Read HMS schema
    schema = pq.read_schema(fpath)
    cols   = [f.name for f in schema]
    print(f"  Columns: {cols}")

    meta = pq.read_metadata(fpath)
    print(f"  Total rows: {meta.num_rows:,}")

    hh_col   = next((c for c in ["household_code", "panid", "household_id"] if c in cols), None)
    upc_col  = next((c for c in ["upc", "upc_code"] if c in cols), None)
    wk_col   = next((c for c in ["week_end", "week", "purchase_date"] if c in cols), None)
    qty_col  = next((c for c in ["quantity", "units", "qty", "hh_purchase_qty"] if c in cols), None)
    w5_col   = next((c for c in ["W5", "w5", "promo_flag", "promo", "promo_coincident_flag"] if c in cols), None)
    depth_col= next((c for c in ["depth_bin", "discount_depth"] if c in cols), None)

    read_cols = [hh_col, upc_col, wk_col, qty_col, w5_col, depth_col, "log_units", "interpurchase_weeks"]
    read_cols = [c for c in dict.fromkeys(read_cols) if c and c in cols]

    # Load only the columns required for the fixed-effects models.
    df = pd.read_parquet(fpath, columns=read_cols)
    print(f"  Loaded: {len(df):,} rows")
    print(f"  Sample:\n{df.head(3)}")

    if not hh_col:
        print("  ERROR: no household identifier found. Check HMS schema.")
        continue
    if not qty_col:
        print("  ERROR: no quantity column found.")
        continue

    # Outcome: log(1 + qty)
    if "log_units" in df.columns:
        df["ln_qty"] = df["log_units"]
    else:
        df["ln_qty"] = np.log1p(df[qty_col].fillna(0))

    # FE identifiers
    df["hh_fe"]    = df[hh_col].astype(str)
    df["hh_upc"]   = df[hh_col].astype(str) + "_" + df[upc_col].astype(str)
    df["week_fe"]  = df[wk_col].astype(str)

    if "interpurchase_weeks" in df.columns:
        df["ipi"] = df["interpurchase_weeks"]
    else:
        df = df.sort_values([hh_col, upc_col, wk_col])
        df["prev_week"] = df.groupby([hh_col, upc_col])[wk_col].shift(1)
        df["ipi"] = (pd.to_datetime(df[wk_col]) -
                     pd.to_datetime(df["prev_week"])).dt.days / 7   # in weeks

    # Depth bin dummies
    bins = ["D2", "D3", "D4", "D5", "D6", "D7"]
    if depth_col == "depth_bin":
        for b in bins:
            df[b] = (df["depth_bin"] == b).astype(np.int8)
        depth_rhs = " + ".join(bins)
    else:
        depth_rhs = None

    # ── Model A: Household FE + treatment ─────────────────────────────────
    if w5_col and df[w5_col].sum() > 100:
        formula_A = f"ln_qty ~ {w5_col} | hh_fe + week_fe"
        try:
            fit_A = pf.feols(formula_A, data=df, vcov={"CRV1": hh_col})
            c_A = fit_A.coef().get(w5_col, np.nan)
            s_A = fit_A.se().get(w5_col, np.nan)
            print(f"  Model A (hh FE + W5): coef={c_A:.4f} SE={s_A:.4f}")
            all_results.append({
                "category": cat, "model": "A_hh_fe", "variable": w5_col,
                "coef": round(c_A, 4), "se": round(s_A, 4),
                "n_obs": len(df),
            })
        except Exception as e:
            print(f"  Model A error: {e}")

    # ── Model B: Household × UPC FE + depth bins ──────────────────────────
    if depth_rhs:
        formula_B = f"ln_qty ~ {depth_rhs} | hh_upc + week_fe"
        try:
            fit_B = pf.feols(formula_B, data=df, vcov={"CRV1": hh_col})
            for b in bins:
                c_b = fit_B.coef().get(b, np.nan)
                s_b = fit_B.se().get(b, np.nan)
                all_results.append({
                    "category": cat, "model": "B_hh_upc_fe", "variable": b,
                    "coef": round(c_b, 4), "se": round(s_b, 4),
                    "n_obs": len(df),
                })
            d7 = fit_B.coef().get("D7", np.nan)
            print(f"  Model B (hh-upc FE + depth): D7 coef={d7:.4f}")
        except Exception as e:
            print(f"  Model B error: {e}")
    elif w5_col and df[w5_col].sum() > 100:
        formula_B = f"ln_qty ~ {w5_col} | hh_upc + week_fe"
        try:
            fit_B = pf.feols(formula_B, data=df, vcov={"CRV1": hh_col})
            c_B = fit_B.coef().get(w5_col, np.nan)
            s_B = fit_B.se().get(w5_col, np.nan)
            print(f"  Model B (hh-upc FE + W5): coef={c_B:.4f} SE={s_B:.4f}")
            all_results.append({
                "category": cat, "model": "B_hh_upc_fe_binary", "variable": w5_col,
                "coef": round(c_B, 4), "se": round(s_B, 4),
                "n_obs": len(df),
            })
        except Exception as e:
            print(f"  Model B error: {e}")

    # ── Model C: Household × UPC FE + W5 + IPI control ───────────────────
    df_c = df[df["ipi"].notna() & df["ipi"].between(0, 52)].copy()
    if w5_col and len(df_c) > 500:
        formula_C = f"ln_qty ~ {w5_col} + ipi | hh_upc + week_fe"
        try:
            fit_C = pf.feols(formula_C, data=df_c, vcov={"CRV1": hh_col})
            c_w5  = fit_C.coef().get(w5_col, np.nan)
            c_ipi = fit_C.coef().get("ipi", np.nan)
            print(f"  Model C (hh-upc FE + W5 + IPI): W5={c_w5:.4f}, IPI={c_ipi:.4f}")
            for var, c_v in [(w5_col, c_w5), ("ipi", c_ipi)]:
                all_results.append({
                    "category": cat, "model": "C_hh_upc_ipi", "variable": var,
                    "coef": round(c_v, 4), "se": round(fit_C.se().get(var, np.nan), 4),
                    "n_obs": len(df_c),
                })
        except Exception as e:
            print(f"  Model C error: {e}")

result_df = pd.DataFrame(all_results)
out_path  = os.path.join(OUT_DIR, "task08_household_fe_results.csv")
result_df.to_csv(out_path, index=False)
print(f"\nResults saved: {out_path}")
print(result_df.to_string(index=False))
print("\nTask 08 complete.")
