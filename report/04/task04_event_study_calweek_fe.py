"""
task04_event_study_calweek_fe.py
=================================
PURPOSE
-------
Add calendar-week fixed effects to the promotion-end event study.

This is the most important remaining robustness check for the offset ratio.
Without calendar-week FE, seasonal demand patterns (e.g. cereal spikes in
September back-to-school, coffee dips in summer) can confound the post-promotion
dip if promotions are systematically timed. Adding week-of-year FE controls for
demand seasonality that would otherwise contaminate the offset estimate.

ESTIMATING EQUATION (with calendar-week FE)
--------------------------------------------
ln(units)_{ist} = sum_tau [delta_tau * 1(t - t*_{is} = tau)] * 1(promo_event_{is}=1)
                + alpha_{is}      [store x UPC FE]
                + alpha_{t}       [calendar-week-of-year FE]  <-- NEW
                + epsilon_{ist}

Where t* is the promotion-end week for promotion spell s at store i.

APPROACH
---------
Because loading 1.4B rows is memory-intensive, this script:
1. Reads the existing event study results (already computed without calweek FE)
   as the baseline.
2. Loads the consolidated parquet in manageable 500K-row batches.
3. Constructs a promotion-end-anchored event study window around each
   promo_event==1 transition, then estimates via pyfixest with week_of_year FE.
4. Compares offset ratios with vs. without calendar-week FE.

The script defaults to a 10% random sample of store-UPC pairs to run on a
laptop; set SAMPLE_FRAC=1.0 for full panel.

MEMORY NOTE
-----------
At SAMPLE_FRAC=0.10, each category loads ~140M rows — manageable with 32GB RAM.
At SAMPLE_FRAC=0.02 (default), each category loads ~28M rows — safe for 16GB.
"""

import os, random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyfixest as pf

# ── Paths ──────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
STAGE5  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage5_event_study\rms_fe"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task04_calweek_fe"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES  = ["cereal", "coffee", "snacks"]
SAMPLE_FRAC = 0.02    # fraction of store-UPC pairs to sample (0.02 = ~28M rows/cat)
PRE_WINDOW  = 8       # event-time bins before promotion end
POST_WINDOW = 8       # event-time bins after promotion end
RANDOM_SEED = 42

# ── Helper: load a sampled category panel ─────────────────────────────────

def load_sampled_panel(cat: str, frac: float) -> pd.DataFrame:
    """
    Load a random fraction of store-UPC pairs from the category parquet.
    Uses PyArrow batch reading to avoid loading all rows into memory first.
    """
    fpath = os.path.join(STAGE2, f"rms_{cat}_analysis.parquet")
    print(f"  Sampling {frac*100:.0f}% of {cat} panel from {fpath}")

    # Phase 1: collect unique store_upc pairs via batch scan
    pf_file = pq.ParquetFile(fpath)
    pairs = set()
    for batch in pf_file.iter_batches(batch_size=200_000,
                                       columns=["store_code_uc", "upc"]):
        pdf = batch.to_pandas()
        pairs.update(zip(pdf["store_code_uc"], pdf["upc"]))

    # Sample fraction
    rng = random.Random(RANDOM_SEED)
    sampled = set(rng.sample(sorted(pairs), k=max(1, int(len(pairs) * frac))))
    print(f"  Unique store-UPC pairs: {len(pairs):,}  ->  sampled: {len(sampled):,}")

    # Phase 2: reload, keeping only sampled pairs
    batches = []
    for batch in pf_file.iter_batches(batch_size=200_000):
        pdf = batch.to_pandas()
        mask = [tuple(r) in sampled
                for r in zip(pdf["store_code_uc"], pdf["upc"])]
        batches.append(pdf[mask])

    panel = pd.concat(batches, ignore_index=True)
    print(f"  Sampled panel rows: {len(panel):,}")
    return panel


# ── Helper: build event-time index ────────────────────────────────────────

def build_event_index(panel: pd.DataFrame, pre: int, post: int) -> pd.DataFrame:
    """
    For each store-UPC pair, find promo_event transition points (0->1),
    label them as promotion ends, and assign event-time tau = t - t_end.
    Keep only rows where -pre <= tau <= post.
    """
    panel = panel.copy().sort_values(["store_code_uc", "upc", "week_end"])
    panel["promo_lag"] = panel.groupby(["store_code_uc", "upc"])["promo_event"].shift(1)
    panel["promo_start"] = ((panel["promo_event"] == 1) &
                            (panel["promo_lag"].fillna(0) == 0)).astype(int)

    # Assign event ID to each promotion spell
    panel["event_id"] = panel.groupby(["store_code_uc", "upc"])["promo_start"].cumsum()
    panel.loc[panel["promo_event"] == 0, "event_id"] = np.nan

    # Keep only non-promoted weeks with last event_id
    # Simplified: mark promotion-end as the last week of each spell
    # Use shift to detect transitions 1->0 or end of promotion
    panel["promo_next"] = panel.groupby(["store_code_uc", "upc"])["promo_event"].shift(-1)
    panel["promo_end"] = ((panel["promo_event"] == 1) &
                          ((panel["promo_next"] == 0) | panel["promo_next"].isna()))
    # t_end_week for each spell
    panel["t_end_week"] = np.where(panel["promo_end"], panel["week_end"], np.nan)
    panel["t_end_week"] = panel.groupby(["store_code_uc", "upc"])["t_end_week"].fillna(method="bfill")

    # Convert week_end to integer week number for arithmetic
    panel["week_int"] = pd.to_datetime(panel["week_end"]).dt.isocalendar().week.astype(int)
    panel["t_end_int"] = pd.to_datetime(panel["t_end_week"]).dt.isocalendar().week.astype(int)

    # Year-adjusted week number to handle year boundaries
    panel["year_week"] = pd.to_datetime(panel["week_end"]).dt.year * 53 + panel["week_int"]
    panel["t_end_year_week"] = pd.to_datetime(panel["t_end_week"]).dt.year * 53 + panel["t_end_int"]
    panel["event_time"] = panel["year_week"] - panel["t_end_year_week"]

    # Keep window
    panel = panel[(panel["event_time"] >= -pre) & (panel["event_time"] <= post)].copy()
    # Reference: event_time = -1 (week before promotion end)
    return panel


# ── Main estimation loop ───────────────────────────────────────────────────

print("=" * 65)
print("Task 04: Event Study with Calendar-Week Fixed Effects")
print("=" * 65)

results = []

for cat in CATEGORIES:
    print(f"\n{'='*40}\n{cat.upper()}")

    # Load baseline (no calweek FE) results for comparison
    baseline_path = os.path.join(STAGE5, f"event_study_fe_{cat}.csv")
    offset_path   = os.path.join(STAGE5, f"offset_share_{cat}.csv")
    if os.path.exists(baseline_path):
        baseline = pd.read_csv(baseline_path)
        offset_base = pd.read_csv(offset_path)["offset_share"].iloc[0]
        print(f"  Baseline offset share (no calweek FE): {offset_base:.4f}")
    else:
        baseline = None
        offset_base = np.nan
        print(f"  WARNING: no baseline event study found at {baseline_path}")

    # Load sampled panel
    panel = load_sampled_panel(cat, SAMPLE_FRAC)

    if "log_units" not in panel.columns and "units" in panel.columns:
        panel["log_units"] = np.log1p(panel["units"])

    # Build event index
    print("  Building event-time index...")
    evpanel = build_event_index(panel, PRE_WINDOW, POST_WINDOW)

    if len(evpanel) < 1000:
        print("  WARNING: too few event-time observations. Skipping.")
        continue

    print(f"  Event-panel rows: {len(evpanel):,}")

    # Construct FE variables
    evpanel["store_upc"]  = evpanel["store_code_uc"].astype(str) + "_" + evpanel["upc"].astype(str)
    evpanel["store_week"] = evpanel["store_code_uc"].astype(str) + "_" + evpanel["week_end"].astype(str)
    evpanel["upc_week"]   = evpanel["upc"].astype(str) + "_" + evpanel["week_end"].astype(str)

    # Event-time dummies (reference = -1)
    tau_vals = list(range(-PRE_WINDOW, POST_WINDOW + 1))
    tau_vals.remove(-1)   # reference
    for tau in tau_vals:
        evpanel[f"tau_{tau:+d}".replace("+", "p").replace("-", "m")] = (
            (evpanel["event_time"] == tau).astype(int)
        )

    tau_names = [f"tau_{tau:+d}".replace("+", "p").replace("-", "m") for tau in tau_vals]
    rhs = " + ".join(tau_names)

    # ── Model A: no calweek FE (replication of stage5) ────────────────
    formula_A = f"log_units ~ {rhs} | store_upc + store_week + upc_week"
    print(f"  Estimating Model A (no calweek FE)...")
    try:
        fit_A = pf.feols(formula_A, data=evpanel, vcov={"CRV1": "store_code_uc"})
    except Exception as e:
        print(f"  ERROR in Model A: {e}")
        continue

    # ── Model B: add week_of_year FE ──────────────────────────────────
    formula_B = (f"log_units ~ {rhs} "
                 f"| store_upc + store_week + upc_week + week_of_year")
    print(f"  Estimating Model B (with calendar-week FE)...")
    try:
        fit_B = pf.feols(formula_B, data=evpanel, vcov={"CRV1": "store_code_uc"})
    except Exception as e:
        print(f"  ERROR in Model B: {e}")
        continue

    # ── Compute offset ratios ──────────────────────────────────────────
    def get_offset(fit, pre=4, post=8):
        """
        Offset = |sum(coef for tau in 1..post)| / sum(coef for tau in -pre..-1 + 0)
        Using the pre-end run-up as denominator.
        Returns (during_sum, post_sum, offset_ratio).
        """
        coefs = fit.coef()
        # During: tau = -pre..-1 (pre-promotion run-up) + tau = 0 (end week)
        during_keys  = [f"tau_m{abs(t)}" if t < 0 else f"tau_p{t}" for t in range(-pre, 1)]
        post_keys    = [f"tau_p{t}" for t in range(1, post + 1)]
        during_sum   = sum(coefs.get(k, 0.0) for k in during_keys)
        post_sum     = sum(coefs.get(k, 0.0) for k in post_keys)
        offset_ratio = (abs(post_sum) / during_sum) if during_sum > 0 else np.nan
        return during_sum, post_sum, offset_ratio

    dur_A, post_A, off_A = get_offset(fit_A)
    dur_B, post_B, off_B = get_offset(fit_B)

    print(f"\n  RESULTS ({cat}):")
    print(f"  Model A (no calweek FE): during={dur_A:.4f}  post={post_A:.4f}  offset={off_A:.4f}")
    print(f"  Model B (calweek FE):    during={dur_B:.4f}  post={post_B:.4f}  offset={off_B:.4f}")
    print(f"  Baseline (full panel):   offset={offset_base:.4f}")
    print(f"  Change A->B: {off_B - off_A:+.4f}")

    # Save event-study coefficients for both models
    for model_label, fit in [("no_calweek", fit_A), ("calweek", fit_B)]:
        rows_ev = []
        coefs_fit = fit.coef()
        se_fit    = fit.se()
        for tau in list(range(-PRE_WINDOW, POST_WINDOW + 1)):
            k = f"tau_m{abs(tau)}" if tau < 0 else ("tau_p0" if tau == 0 else f"tau_p{tau}")
            ref = (tau == -1)
            c   = 0.0 if ref else coefs_fit.get(k, np.nan)
            s   = 0.0 if ref else se_fit.get(k, np.nan)
            rows_ev.append({"event_time": tau, "coef": c, "se": s,
                            "ci_low": c - 1.96*s, "ci_high": c + 1.96*s,
                            "is_reference": ref, "model": model_label})
        ev_df = pd.DataFrame(rows_ev)
        ev_path = os.path.join(OUT_DIR, f"event_study_calweek_{cat}_{model_label}.csv")
        ev_df.to_csv(ev_path, index=False)

    results.append({
        "category": cat,
        "offset_baseline_full": round(offset_base, 4),
        "offset_A_no_calweek":  round(off_A, 4),
        "offset_B_calweek":     round(off_B, 4),
        "change_A_to_B":        round(off_B - off_A, 4),
        "sample_frac":          SAMPLE_FRAC,
    })

summary_df = pd.DataFrame(results)
summary_path = os.path.join(OUT_DIR, "task04_offset_comparison.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved: {summary_path}")
print(summary_df.to_string(index=False))
print("\nTask 04 complete.")
