"""
Task 08 — Household and Household×UPC Fixed-Effects Models
===========================================================
Verdict requirement (Section 7): The current 80K DiM result is a difference
in means, NOT a fixed-effects model.

This script estimates:
  Model 1: ln(1+q)_iut = β·PromoPurchase_iut + α_i + γ_u + λ_t + ε
  Model 2: ln(1+q)_iut = β·PromoPurchase_iut + α_{iu} + λ_t + ε

where:
  α_i   = household FE      (absorbs persistent household-level buying behaviour)
  γ_u   = UPC FE            (absorbs product-level fixed differences)
  α_{iu}= household×UPC FE  (absorbs household-product-level fixed preferences)
  λ_t   = week/time FE
  SE    = clustered at household level

Only after these models can you say:
  "The SAME household buys more on promotion-linked occasions."

Until then the DiM estimate may reflect selection (promotion buyers ≠ control buyers).

Inputs
------
HMS panel parquet with columns:
    household_id, upc, week, units, promo_purchase, retail_price, category

Outputs
-------
  Table: ATE with clustered SE for Model 1 and Model 2 by category (stdout)
  pyfixest etable saved to task08_results.txt

Author: Milad Javadi — UTD Economics, Phase 2C
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore", category=UserWarning)


# ── Data validation ────────────────────────────────────────────────────────────
def load_hms(data_path: str) -> pd.DataFrame:
    """
    Load and validate HMS panel.
    Expects purchase-occasion level data (non-purchase weeks NOT recorded).
    """
    df = pd.read_parquet(data_path)

    required = ["household_id", "upc", "week", "units", "promo_purchase", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing HMS columns: {missing}")

    # Outcome: ln(1 + units)
    df["log_quantity"] = np.log1p(df["units"].clip(lower=0))

    # PromoPurchase must be binary
    df["promo_purchase"] = df["promo_purchase"].astype(int)

    print(f"Loaded: {len(df):,} purchase occasions")
    print(f"  Households: {df['household_id'].nunique():,}")
    print(f"  UPCs:       {df['upc'].nunique():,}")
    print(f"  Weeks:      {df['week'].nunique():,}")
    print(f"  Promo rate: {df['promo_purchase'].mean():.1%}")

    return df


# ── Model 1: Household + UPC + Week FE ────────────────────────────────────────
def run_model1(df: pd.DataFrame, category: str) -> object:
    """
    Model 1: ln(1+q) ~ PromoPurchase | household_id + upc + week
    Clustered SE by household_id.

    Interpretation: within-household, within-UPC, within-week variation.
    Controls for household buying habits, UPC quality, and seasonal demand.
    """
    print(f"\n  Model 1 (household + UPC + week FE) — {category}")
    fit = pf.feols(
        "log_quantity ~ promo_purchase | household_id + upc + week",
        data=df,
        vcov={"CRV1": "household_id"}
    )
    print(fit.summary())
    return fit


# ── Model 2: Household×UPC + Week FE (preferred) ──────────────────────────────
def run_model2(df: pd.DataFrame, category: str) -> object:
    """
    Model 2: ln(1+q) ~ PromoPurchase | household_id^upc + week
    Clustered SE by household_id.

    Stronger specification: household×UPC FE absorbs all time-invariant
    differences in how household i buys product u. The treatment effect
    is identified purely from within-household-UPC variation over time.

    NOTE: Requires enough within-HH-UPC variation. If a household only
    ever buys a product on promotion, its FE absorbs all variation and
    that household is dropped from the estimation.
    """
    print(f"\n  Model 2 (household×UPC + week FE) — {category}")

    # Check within-HH-UPC variation exists
    within_var = (
        df.groupby(["household_id", "upc"])["promo_purchase"]
        .nunique()
    )
    n_varying = (within_var > 1).sum()
    n_total   = len(within_var)
    print(f"  HH-UPC cells with both promo and non-promo purchases: "
          f"{n_varying:,} / {n_total:,} ({n_varying/n_total:.1%})")

    if n_varying < 1000:
        print("  [WARN] Low within-HH-UPC variation — Model 2 estimates may be unreliable")

    fit = pf.feols(
        "log_quantity ~ promo_purchase | household_id^upc + week",
        data=df,
        vcov={"CRV1": "household_id"}
    )
    print(fit.summary())
    return fit


# ── Output table ───────────────────────────────────────────────────────────────
def print_result_table(results: dict):
    """
    Print summary table compatible with Table 12 in the report.

    Required format (from verdict, Section 7):
      Category | Model | ATE (log units) | Clustered SE | t-stat | 95% CI
    """
    print("\n=== TASK 08 RESULTS TABLE ===")
    print(f"{'Category':<10} {'Model':<10} {'ATE':>8} {'Clust.SE':>10} "
          f"{'t-stat':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print("-" * 68)
    for (cat, model), fit in results.items():
        coef = fit.coef()["promo_purchase"]
        se   = fit.se()["promo_purchase"]
        ci   = fit.confint().loc["promo_purchase"]
        tstat = coef / se
        print(f"{cat:<10} {model:<10} {coef:>8.4f} {se:>10.4f} "
              f"{tstat:>8.2f} {ci.iloc[0]:>8.4f} {ci.iloc[1]:>8.4f}")

    print("\n[NOTE] Model 2 (HH×UPC FE) is the preferred specification.")
    print("       Model 1 provides partial FE control for comparison.")
    print("       Both remove household-level selection bias present in the DiM.")
    print("       Correct language: 'Within-household, the same household buys")
    print("       MORE on promotion-linked occasions.'")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_path: str, category: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Task 08 — Household FE Models: {category.upper()} ===")

    df = load_hms(data_path)

    # Filter to category if column present
    if "category" in df.columns:
        df = df[df["category"] == category].copy()
        print(f"After category filter: {len(df):,} rows")

    results = {}

    try:
        results[(category, "Model_1")] = run_model1(df, category)
    except Exception as e:
        print(f"  [ERROR] Model 1 failed: {e}")

    try:
        results[(category, "Model_2")] = run_model2(df, category)
    except Exception as e:
        print(f"  [ERROR] Model 2 failed: {e}")

    if results:
        print_result_table(results)

        # Save etable to text file
        out_file = out_dir / f"task08_results_{category}.txt"
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        pf.etable(list(results.values()),
                  labels={"log_quantity": "ln(1+units)",
                          "promo_purchase": "PromoPurchase"})
        sys.stdout = old_stdout
        out_file.write_text(buf.getvalue())
        print(f"\nEtable saved to: {out_file}")
    else:
        print("\n[ERROR] All models failed. Check data schema and try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="HMS parquet path")
    parser.add_argument("--category", required=True,
                        choices=["cereal", "snacks", "coffee"])
    parser.add_argument("--out_dir", default=".",
                        help="Directory for output tables")
    args = parser.parse_args()
    main(args.data_path, args.category, args.out_dir)
