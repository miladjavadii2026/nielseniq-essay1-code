"""
Task 10 — UPC-Exit Clearance Robustness Test
============================================
Verdict requirement (Section 7): The D5–D7 reversal chart only RESTATES
the reversal. It does NOT test clearance.

This script tests whether the D6/D7 depth-response reversals in coffee and
snacks are driven by clearance discounting near UPC exit:

  1. Compute WeeksToExit_u,t = LastObservedWeek_u - t
  2. Report share of D6/D7 observations within 4/8/12 weeks of exit
  3. Re-estimate the seven-bin FE depth model excluding near-exit observations
  4. Compare D6/D7 coefficients: original vs. excl. 4/8/12 final weeks

Key result to test:
  H_clearance: If coffee/snack D6–D7 reversals DISAPPEAR after removing
               near-exit obs, clearance contamination is confirmed.
  If reversals PERSIST, they reflect genuine price-elasticity saturation.

Required output table (from verdict, Section 7):
  Category | Original D7 | Excl. 4 wks | Excl. 8 wks | Excl. 12 wks

Correct language AFTER this script runs:
  IF reversal disappears:
    "Coffee D7 reversal is clearance-driven: coefficient rises from 0.591
     to [INSERT] after excluding final-12-week observations."
  IF reversal persists:
    "The D7 reversal is not explained by clearance discounting near UPC exit."

Inputs
------
RMS panel parquet with columns:
    store_id, upc, week, log_quantity, depth_bin, promo_flag

Outputs
-------
  Table: D6/D7 coefficients by exclusion window (stdout + CSV)
  Table: share of D6/D7 obs in near-exit periods (stdout)
  Figure: Task10_ClearanceRobustness.png — coefficient comparison

Author: Milad Javadi — UTD Economics, Phase 2C
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfixest as pf

warnings.filterwarnings("ignore", category=UserWarning)


# ── Step 1: Compute weeks-to-exit for each UPC ────────────────────────────────
def add_weeks_to_exit(df: pd.DataFrame) -> pd.DataFrame:
    """
    WeeksToExit_u,t = LastObservedWeek_u - t

    This measures how many weeks remain before the UPC disappears from the panel.
    A UPC with WeeksToExit = 3 at observation t has only 3 more weeks of sales.

    NOTE: This uses the LAST OBSERVED WEEK as a proxy for exit. This will
    include UPCs that temporarily drop out (e.g., out-of-stock). For a cleaner
    exit proxy, ideally exclude UPCs that reappear after a gap.
    """
    last_week = df.groupby("upc")["week"].transform("max")
    df["weeks_to_exit"] = last_week - df["week"]
    return df


# ── Step 2: Share of D6/D7 obs near exit ──────────────────────────────────────
def print_near_exit_shares(df: pd.DataFrame, category: str):
    """
    Report the share of D6 and D7 observations falling within 4/8/12 weeks
    of UPC exit. High shares = clearance contamination plausible.
    """
    print(f"\n  Near-exit shares for {category}:")
    print(f"  {'Bin':<5} {'All obs':>8} {'In 4wk':>8} {'In 8wk':>8} {'In 12wk':>8}")
    print("  " + "-" * 45)

    for bin_label in ["D6", "D7"]:
        mask = df["depth_bin"] == bin_label
        n_total = mask.sum()
        if n_total == 0:
            print(f"  {bin_label:<5} No observations")
            continue
        n4  = (mask & (df["weeks_to_exit"] <= 4)).sum()
        n8  = (mask & (df["weeks_to_exit"] <= 8)).sum()
        n12 = (mask & (df["weeks_to_exit"] <= 12)).sum()
        print(f"  {bin_label:<5} {n_total:>8,} {n4/n_total:>7.1%} "
              f"{n8/n_total:>7.1%} {n12/n_total:>7.1%}")


# ── Step 3: FE model with near-exit exclusions ────────────────────────────────
def run_fe_with_exclusion(df: pd.DataFrame, exclude_weeks: int | None) -> dict:
    """
    Run seven-bin FE depth model on sample with optional exclusion of near-exit obs.

    Args:
        df: full RMS panel with depth_bin and weeks_to_exit
        exclude_weeks: if not None, exclude obs with weeks_to_exit <= this value
    """
    if exclude_weeks is not None:
        sample = df[df["weeks_to_exit"] > exclude_weeks].copy()
        label = f"excl_final_{exclude_weeks}wk"
    else:
        sample = df.copy()
        label = "original"

    n_obs    = len(sample)
    n_d6_obs = (sample["depth_bin"] == "D6").sum()
    n_d7_obs = (sample["depth_bin"] == "D7").sum()

    formula = "log_quantity ~ i(depth_bin, ref='D1') | store_id^upc + store_id^week + upc^week"

    try:
        fit = pf.feols(formula, data=sample, vcov={"CRV1": "store_id"})
        coefs = fit.coef()
        ci    = fit.confint()

        d6_key = "depth_bin::D6"
        d7_key = "depth_bin::D7"

        d6_est = coefs.get(d6_key, np.nan)
        d7_est = coefs.get(d7_key, np.nan)
        d6_lo  = ci.loc[d6_key, "2.5%"]  if d6_key in ci.index else np.nan
        d6_hi  = ci.loc[d6_key, "97.5%"] if d6_key in ci.index else np.nan
        d7_lo  = ci.loc[d7_key, "2.5%"]  if d7_key in ci.index else np.nan
        d7_hi  = ci.loc[d7_key, "97.5%"] if d7_key in ci.index else np.nan

        return {
            "label": label, "n_obs": n_obs,
            "n_d6": n_d6_obs, "n_d7": n_d7_obs,
            "D6": d6_est, "D6_lo": d6_lo, "D6_hi": d6_hi,
            "D7": d7_est, "D7_lo": d7_lo, "D7_hi": d7_hi,
        }
    except Exception as e:
        print(f"  [ERROR] FE model ({label}) failed: {e}")
        return {"label": label, "n_obs": n_obs, "D6": np.nan, "D7": np.nan}


# ── Step 4: Plot coefficient comparison ───────────────────────────────────────
def plot_clearance_comparison(results: list, category: str, fig_dir: Path):
    """
    Bar chart: D7 coefficient (with CI) across exclusion windows.
    Clear visual test: if D7 rises as near-exit obs are removed → clearance.
    """
    labels = [r["label"].replace("_", " ") for r in results]
    d6_vals = [r["D6"] for r in results]
    d7_vals = [r["D7"] for r in results]
    d6_lo   = [r["D6_lo"] for r in results]
    d6_hi   = [r["D6_hi"] for r in results]
    d7_lo   = [r["D7_lo"] for r in results]
    d7_hi   = [r["D7_hi"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, vals, los, his, bin_label, color in [
        (axes[0], d6_vals, d6_lo, d6_hi, "D6 (40–50%)", "#e67e22"),
        (axes[1], d7_vals, d7_lo, d7_hi, "D7 (≥50%)",   "#c0392b"),
    ]:
        errs_lo = [max(v - lo, 0) for v, lo in zip(vals, los)]
        errs_hi = [max(hi - v, 0) for v, hi in zip(vals, his)]
        bars = ax.bar(x, vals, color=color, alpha=0.75, edgecolor="white",
                      yerr=[errs_lo, errs_hi], capsize=4, error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"{bin_label} FE estimate [ln(1+units)]", fontsize=9)
        ax.set_title(f"{bin_label} coefficient across exclusion windows\n"
                     f"Rising coefficient → clearance contamination", fontsize=9)
        ax.axhline(vals[0] if vals[0] == vals[0] else 0, color="gray",
                   lw=1, linestyle="--", alpha=0.5, label="Original")

    fig.suptitle(
        f"Task 10: Clearance Robustness — {category.title()} (NielsenIQ)\n"
        f"Bin estimates after removing observations within N weeks of UPC exit",
        fontsize=10
    )
    plt.tight_layout()
    out = fig_dir / "Task10_ClearanceRobustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_path: str, category: str, fig_dir: str, out_dir: str):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Task 10 — UPC-Exit Clearance Test: {category.upper()} ===")

    df = pd.read_parquet(data_path)
    df["log_quantity"] = np.log1p(df["units"].clip(lower=0))
    if "category" in df.columns:
        df = df[df["category"] == category].copy()

    print(f"Loaded {len(df):,} obs | {df['upc'].nunique():,} UPCs")

    print("\nStep 1: Computing weeks to exit...")
    df = add_weeks_to_exit(df)

    print("\nStep 2: Near-exit shares by depth bin...")
    print_near_exit_shares(df, category)

    print("\nStep 3: FE models across exclusion windows...")
    results = []
    for excl in [None, 4, 8, 12]:
        r = run_fe_with_exclusion(df, excl)
        results.append(r)
        print(f"  {r['label']:<25} N={r['n_obs']:>12,} | "
              f"D6={r['D6']:.3f} | D7={r['D7']:.3f}")

    print("\nStep 4: Plotting...")
    plot_clearance_comparison(results, category, fig_dir)

    # Summary table
    print("\n=== TASK 10 RESULTS TABLE ===")
    print(f"  {'Window':<22} {'D6 est':>9} {'D7 est':>9}")
    print("  " + "-" * 44)
    for r in results:
        print(f"  {r['label']:<22} {r['D6']:>9.3f} {r['D7']:>9.3f}")

    # Save CSV
    out_csv = out_dir / f"task10_clearance_{category}.csv"
    pd.DataFrame(results)[["label", "n_obs", "D6", "D6_lo", "D6_hi",
                             "D7", "D7_lo", "D7_hi"]].to_csv(out_csv, index=False)
    print(f"\n  Results saved: {out_csv}")

    # Interpretation guidance
    d7_orig = results[0]["D7"]
    d7_excl12 = results[3]["D7"]
    if not np.isnan(d7_orig) and not np.isnan(d7_excl12):
        delta = d7_excl12 - d7_orig
        if delta > 0.05:
            print(f"\n  [FINDING] D7 rises by {delta:.3f} after removing near-exit obs.")
            print(f"  Consistent with clearance contamination. Correct language:")
            print(f"  'The D7 reversal is partially explained by clearance discounting.'")
        else:
            print(f"\n  [FINDING] D7 changes by only {delta:.3f} after exclusion.")
            print(f"  Clearance contamination does NOT explain the reversal.")
            print(f"  Correct language: 'The D7 reversal persists after removing")
            print(f"  near-exit observations, suggesting genuine saturation.'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--category",  required=True,
                        choices=["cereal", "snacks", "coffee"])
    parser.add_argument("--fig_dir",   default=".")
    parser.add_argument("--out_dir",   default=".")
    args = parser.parse_args()
    main(args.data_path, args.category, args.fig_dir, args.out_dir)
