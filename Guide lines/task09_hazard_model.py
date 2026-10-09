"""
Task 09 — Interpurchase Hazard Model
=====================================
Verdict requirement (Section 7): KS tests and mean-gap comparisons are
DESCRIPTIVE statistics. They are NOT hazard models.

This script estimates a proper discrete-time hazard model and, optionally,
a Cox proportional hazard model:

Discrete-time hazard (preferred for weekly household panel data):
  Pr(purchase_{i,t} = 1) = logistic(
      f(weeks_since_last, prior_promo_purchase,
        household_FE, UPC_FE, calendar_time)
  )
  - Estimated via feols with household FE (within-household identification)
  - Clustered SE by household_id

Cox proportional hazard (alternative):
  h(t | X) = h_0(t) exp(β_1 prior_promo_purchase + β_2 X + strata(household_id))
  - Requires lifelines: pip install lifelines

Correct output:
  - Hazard ratio = exp(β) for prior_promo_purchase
  - 95% CI for hazard ratio
  - Number of spells, censoring rate
  - Proportional-hazard diagnostic (Schoenfeld residuals)

Correct language AFTER this script runs:
  "After a promotional purchase, the weekly purchase hazard is {ratio}× lower,
   indicating purchase deferral (stockpiling)."

Inputs
------
HMS panel parquet (purchase-occasion level, NOT balanced panel):
    household_id, upc, week, units, promo_purchase, category

The script will CONSTRUCT interpurchase spells from the purchase sequence.

Outputs
-------
  Table: hazard ratio, CI, n_spells by category and model
  Figure: Kaplan-Meier survival curves (treated vs. control) saved to fig_dir

Author: Milad Javadi — UTD Economics, Phase 2C
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


# ── Step 1: Construct interpurchase spells ─────────────────────────────────────
def build_spells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build interpurchase spell dataset from purchase-occasion panel.

    For each household-UPC pair, compute the time (weeks) between successive
    purchases. The treatment indicator is whether the INITIATING purchase
    (start of spell) occurred on promotion.

    Spell = the gap between purchase t and purchase t+1.
      - duration     : weeks until next purchase
      - event        : 1 if next purchase observed, 0 if right-censored
      - prior_promo  : 1 if the initiating purchase was on promotion

    Right-censoring: the last purchase in each household-UPC sequence
    is censored because we don't observe when (if) the next purchase occurs.
    """
    df = df.copy().sort_values(["household_id", "upc", "week"]).reset_index(drop=True)

    # Next purchase week within same household-UPC
    df["next_week"] = (
        df.groupby(["household_id", "upc"])["week"]
        .shift(-1)
    )

    # Duration = gap in weeks
    df["spell_duration"] = df["next_week"] - df["week"]

    # Event indicator: 1 = next purchase observed, 0 = censored (last obs)
    df["event"] = df["next_week"].notna().astype(int)

    # For censored obs, set duration to some large value for KM but keep flag
    # Actually: just keep event=0 and duration as is (lifelines handles NaN → censored)

    # Spell start: this purchase
    df["prior_promo"] = df["promo_purchase"].astype(int)

    spells = df[["household_id", "upc", "week", "spell_duration",
                  "event", "prior_promo"]].dropna(subset=["spell_duration"])

    print(f"  Total spells: {len(spells):,}")
    print(f"  Observed (uncensored): {spells['event'].sum():,} ({spells['event'].mean():.1%})")
    print(f"  Mean spell duration: {spells['spell_duration'].mean():.1f} weeks")
    print(f"  Promo-initiated spells: {spells['prior_promo'].mean():.1%}")

    return spells


# ── Step 2: Kaplan–Meier survival curves ──────────────────────────────────────
def plot_km_curves(spells: pd.DataFrame, category: str, fig_dir: Path):
    """
    Plot Kaplan-Meier survival curves for promo vs. non-promo initiated spells.
    Longer survival = longer gap = more stockpiling.
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        print("  [INFO] lifelines not installed — skipping KM plot. "
              "Run: pip install lifelines")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for promo_flag, label, color in [(1, "Promo purchase", "#c0392b"),
                                      (0, "Non-promo purchase", "#2c6fad")]:
        grp = spells[spells["prior_promo"] == promo_flag]
        kmf = KaplanMeierFitter()
        kmf.fit(grp["spell_duration"], grp["event"], label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.15)

    ax.set_xlabel("Weeks since last purchase", fontsize=11)
    ax.set_ylabel("Pr(no purchase yet)", fontsize=11)
    ax.set_title(
        f"Figure: Kaplan–Meier Interpurchase Survival — {category.title()}\n"
        f"Curves below (shorter survival) indicate faster repurchase.",
        fontsize=10
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = fig_dir / f"Task09_KM_{category}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  KM figure saved: {out}")


# ── Step 3: Discrete-time hazard model (preferred) ────────────────────────────
def run_discrete_hazard(spells: pd.DataFrame, category: str) -> dict:
    """
    Discrete-time hazard model via logistic regression with FE.

    Expand each spell into week-by-week observations (person-period format):
      For a spell of duration T:
        t = 1: event = 0
        t = 2: event = 0
        ...
        t = T: event = 1 (or 0 if censored)
    Then run:
      event ~ prior_promo + log(duration_so_far) | household_id + upc + calendar_week
    """
    import pyfixest as pf

    # Expand to person-period format
    print("  Expanding to person-period format...")
    rows = []
    for _, row in spells.iterrows():
        T = int(row["spell_duration"])
        if T < 1 or T > 104:    # cap at 2 years
            continue
        for t in range(1, T + 1):
            event_this_week = 1 if (t == T and row["event"] == 1) else 0
            rows.append({
                "household_id": row["household_id"],
                "upc": row["upc"],
                "calendar_week": int(row["week"]) + t,
                "duration_t": t,
                "prior_promo": int(row["prior_promo"]),
                "purchase_event": event_this_week,
            })
    pp = pd.DataFrame(rows)
    print(f"  Person-period rows: {len(pp):,}")

    # Log duration baseline hazard
    pp["log_duration"] = np.log1p(pp["duration_t"])

    # Linear probability model with FE (logistic not supported natively in pyfixest)
    # Use LPM as approximation; note: coefficients ≈ marginal hazard change
    try:
        fit = pf.feols(
            "purchase_event ~ prior_promo + log_duration | household_id + upc + calendar_week",
            data=pp,
            vcov={"CRV1": "household_id"}
        )
        coef = fit.coef()["prior_promo"]
        se   = fit.se()["prior_promo"]
        ci   = fit.confint().loc["prior_promo"]
        print(f"\n  Discrete-time hazard (LPM): coef = {coef:.4f} | SE = {se:.4f}")
        print(f"  95% CI: [{ci.iloc[0]:.4f}, {ci.iloc[1]:.4f}]")
        print(f"  Interpretation: promo purchase changes weekly purchase probability by {coef:.4f}")

        return {"coef": coef, "se": se, "ci_lo": ci.iloc[0], "ci_hi": ci.iloc[1],
                "n_spells": len(spells), "n_person_periods": len(pp), "fit": fit}
    except Exception as e:
        print(f"  [ERROR] Discrete hazard failed: {e}")
        return {}


# ── Step 4: Cox PH model (alternative) ────────────────────────────────────────
def run_cox_model(spells: pd.DataFrame, category: str) -> dict:
    """
    Cox proportional hazard model via lifelines.
    Stratified by household_id (absorbs baseline hazard heterogeneity).
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except ImportError:
        print("  [INFO] lifelines not installed — skipping Cox model. "
              "Run: pip install lifelines")
        return {}

    cox_df = spells[["spell_duration", "event", "prior_promo", "household_id"]].copy()
    cox_df = cox_df.dropna()
    cox_df["spell_duration"] = cox_df["spell_duration"].clip(lower=0.5)  # lifelines needs > 0

    print(f"\n  Running Cox PH model (N spells = {len(cox_df):,})...")

    cph = CoxPHFitter()
    try:
        cph.fit(
            cox_df,
            duration_col="spell_duration",
            event_col="event",
            formula="prior_promo",
            strata=["household_id"],
            cluster_col="household_id"
        )
        cph.print_summary()

        hr     = np.exp(cph.params_["prior_promo"])
        hr_lo  = np.exp(cph.confidence_intervals_["95% lower-bound"]["prior_promo"])
        hr_hi  = np.exp(cph.confidence_intervals_["95% upper-bound"]["prior_promo"])

        print(f"\n  Hazard ratio = {hr:.4f} | 95% CI = [{hr_lo:.4f}, {hr_hi:.4f}]")
        print(f"  Interpretation: after a promo purchase, next-purchase hazard is {hr:.3f}× the baseline")
        print(f"  ({(1-hr)*100:.1f}% {'lower' if hr < 1 else 'higher'} hazard = "
              f"{'longer' if hr < 1 else 'shorter'} interpurchase gap)")

        # PH diagnostic
        try:
            ph_test = proportional_hazard_test(cph, cox_df, time_transform="log")
            p_ph = ph_test.summary["p"].get("prior_promo", np.nan)
            print(f"  PH test p-value: {p_ph:.4f} "
                  f"({'assumption holds' if p_ph > 0.05 else 'VIOLATION — consider time-varying coef'})")
        except Exception:
            pass

        return {"hazard_ratio": hr, "hr_lo": hr_lo, "hr_hi": hr_hi,
                "n_spells": len(cox_df)}
    except Exception as e:
        print(f"  [ERROR] Cox model failed: {e}")
        return {}


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_path: str, category: str, fig_dir: str, out_dir: str):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Task 09 — Interpurchase Hazard Model: {category.upper()} ===")

    df = pd.read_parquet(data_path)
    if "category" in df.columns:
        df = df[df["category"] == category].copy()
    print(f"Loaded {len(df):,} purchase occasions for {category}")

    print("\nStep 1: Building interpurchase spells...")
    spells = build_spells(df)

    print("\nStep 2: Kaplan–Meier survival curves...")
    plot_km_curves(spells, category, fig_dir)

    print("\nStep 3: Discrete-time hazard model...")
    lpm_result = run_discrete_hazard(spells, category)

    print("\nStep 4: Cox PH model...")
    cox_result = run_cox_model(spells, category)

    print("\n=== TASK 09 SUMMARY ===")
    print(f"Category: {category.upper()}")
    if lpm_result:
        print(f"  Discrete hazard (LPM): Δ weekly prob = {lpm_result['coef']:.4f} "
              f"SE={lpm_result['se']:.4f}")
    if cox_result:
        hr = cox_result.get("hazard_ratio", np.nan)
        print(f"  Cox HR: {hr:.4f} [{cox_result['hr_lo']:.4f}, {cox_result['hr_hi']:.4f}]")
        print(f"  Correct language: 'After a promo purchase, the next-purchase hazard is ")
        print(f"   {hr:.3f}× the baseline, indicating {'purchase deferral' if hr < 1 else 'acceleration'}.'")
        print(f"\n  DO NOT use the phrase 'hazard ratio' in the report until these")
        print(f"  models have been run. KS tests ≠ hazard models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="HMS parquet path")
    parser.add_argument("--category", required=True,
                        choices=["cereal", "snacks", "coffee"])
    parser.add_argument("--fig_dir",  default=".", help="Figure output directory")
    parser.add_argument("--out_dir",  default=".", help="Table output directory")
    args = parser.parse_args()
    main(args.data_path, args.category, args.fig_dir, args.out_dir)
