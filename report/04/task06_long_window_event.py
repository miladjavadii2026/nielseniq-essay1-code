"""
task06_long_window_event.py
============================
PURPOSE
-------
Extend the post-promotion event window from 8 to 16 weeks and compare the
offset ratio at multiple post-promotion horizons (4, 8, 12, 16 weeks).

WHY THIS MATTERS
-----------------
Coffee's measured 8-week offset (0.48) may understate total intertemporal
substitution because coffee inventory is consumed slowly — households may
draw down purchased stock over 10–16 weeks rather than 8. The current finite-
window offset captures only the portion of payback that falls within 8 weeks
of the promotion end. Extending to 16 weeks tests whether the coffee offset
ratio grows substantially, consistent with the slow-drawdown hypothesis in
the two-axis framework (Section 5A.2 of IS v4).

DESIGN
------
- Extend post-event window to t+16.
- Compute rolling cumulative offset at horizons h = 4, 8, 12, 16 weeks:
  offset(h) = |sum_{t=1}^{h} beta_t| / sum_{t=-4}^{0} beta_t
- All horizons use the same pre-end window (t=-4 to t=0) as denominator.
- If coffee offset grows substantially from h=8 to h=16, this confirms the
  slow-drawdown interpretation and validates IS v4's two-object distinction.
- Uses existing stage5 coefficients where available for the 8-week window;
  re-estimates for h>8 using the sampled panel.

MEMORY NOTE
-----------
SAMPLE_FRAC=0.03 keeps memory under 16GB. Increase for full panel.
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyfixest as pf

# ── Config ─────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
STAGE5  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage5_event_study\rms_fe"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task06_long_window"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES  = ["cereal", "coffee", "snacks"]
SAMPLE_FRAC = 0.03
PRE_WINDOW  = 8
POST_WINDOW = 16       # EXTENDED from 8 to 16
HORIZONS    = [4, 8, 12, 16]
PRE_DENOM   = 4        # bins used in denominator (t=-4 to t=0)
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# ── Helper: load + event-panel construction ────────────────────────────────

def load_and_build_event(cat: str, frac: float) -> pd.DataFrame:
    fpath = os.path.join(STAGE2, f"rms_{cat}_analysis.parquet")
    pf_file = pq.ParquetFile(fpath)

    stores = set()
    for batch in pf_file.iter_batches(batch_size=100_000, columns=["store_code_uc"]):
        stores.update(batch.column("store_code_uc").to_pylist())
    sampled = set(rng.choice(sorted(stores),
                             size=max(1, int(len(stores) * frac)),
                             replace=False))

    batches = []
    for batch in pf_file.iter_batches(batch_size=200_000):
        pdf = batch.to_pandas()
        batches.append(pdf[pdf["store_code_uc"].isin(sampled)])
    panel = pd.concat(batches, ignore_index=True)
    panel = panel.sort_values(["store_code_uc", "upc", "week_end"])

    # Mark promotion ends
    panel["promo_end"] = ((panel["promo_event"] == 1) &
        (panel.groupby(["store_code_uc", "upc"])["promo_event"].shift(-1).fillna(0) == 0))
    panel["t_end_week"] = np.where(panel["promo_end"], panel["week_end"], np.nan)
    panel["t_end_week"] = panel.groupby(["store_code_uc", "upc"])["t_end_week"].fillna(method="bfill")

    for col in ["week_end", "t_end_week"]:
        dt = pd.to_datetime(panel[col])
        panel[f"yw_{col}"] = dt.dt.year * 53 + dt.dt.isocalendar().week.astype(int)
    panel["event_time"] = panel["yw_week_end"] - panel["yw_t_end_week"]

    event = panel[
        (panel["event_time"] >= -PRE_WINDOW) &
        (panel["event_time"] <= POST_WINDOW) &
        panel["t_end_week"].notna()
    ].copy()

    if "log_units" not in event.columns:
        event["log_units"] = np.log1p(event["units"])

    event["store_upc"]  = event["store_code_uc"].astype(str) + "_" + event["upc"].astype(str)
    event["store_week"] = event["store_code_uc"].astype(str) + "_" + event["week_end"].astype(str)
    event["upc_week"]   = event["upc"].astype(str) + "_" + event["week_end"].astype(str)

    taus = [t for t in range(-PRE_WINDOW, POST_WINDOW + 1) if t != -1]
    for tau in taus:
        col = f"tau_{tau:+d}".replace("+", "p").replace("-", "m")
        event[col] = (event["event_time"] == tau).astype(np.int8)

    return event


def rolling_offsets(coefs: dict, pre_denom: int, horizons: list) -> dict:
    """Compute offset ratio at multiple post-promotion horizons."""
    during_keys = ([f"tau_m{abs(t)}" for t in range(-pre_denom, 0)] + ["tau_p0"])
    during_sum  = sum(coefs.get(k, 0.0) for k in during_keys)
    out = {}
    for h in horizons:
        post_keys = [f"tau_p{t}" for t in range(1, h + 1)]
        post_sum  = sum(coefs.get(k, 0.0) for k in post_keys)
        out[f"offset_h{h}"] = abs(post_sum) / during_sum if during_sum > 1e-8 else np.nan
    return out


# ── Main ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("Task 06: Long-Window Post-Promotion Event Study")
print(f"  Post-window: t+{POST_WINDOW}  |  Horizons: {HORIZONS}")
print("=" * 65)

taus     = [t for t in range(-PRE_WINDOW, POST_WINDOW + 1) if t != -1]
tau_names = [f"tau_{t:+d}".replace("+", "p").replace("-", "m") for t in taus]
rhs      = " + ".join(tau_names)
formula  = f"log_units ~ {rhs} | store_upc + store_week + upc_week"

all_results = []
all_coeffs  = []

for cat in CATEGORIES:
    print(f"\n{cat.upper()}")
    event = load_and_build_event(cat, SAMPLE_FRAC)
    print(f"  Event panel: {len(event):,} rows")

    try:
        fit = pf.feols(formula, data=event, vcov={"CRV1": "store_code_uc"})
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    coefs_dict = dict(fit.coef())
    se_dict    = dict(fit.se())

    # Coefficient path for each event_time
    for tau in range(-PRE_WINDOW, POST_WINDOW + 1):
        key = f"tau_m{abs(tau)}" if tau < 0 else ("tau_m0" if tau == 0 else f"tau_p{tau}")
        ref = (tau == -1)
        c   = 0.0 if ref else coefs_dict.get(key, np.nan)
        s   = 0.0 if ref else se_dict.get(key, np.nan)
        all_coeffs.append({
            "category": cat, "event_time": tau,
            "coef": c, "se": s,
            "ci_low": c - 1.96*s, "ci_high": c + 1.96*s,
            "is_reference": ref,
        })

    # Rolling offset ratios
    offs = rolling_offsets(coefs_dict, PRE_DENOM, HORIZONS)
    row  = {"category": cat, **offs, "sample_frac": SAMPLE_FRAC}
    all_results.append(row)

    print(f"  Rolling offsets:")
    for h in HORIZONS:
        print(f"    h={h:2d} weeks: {offs.get(f'offset_h{h}', np.nan):.4f}")

coeff_df  = pd.DataFrame(all_coeffs)
result_df = pd.DataFrame(all_results)

coeff_path  = os.path.join(OUT_DIR, "task06_long_window_coefficients.csv")
result_path = os.path.join(OUT_DIR, "task06_rolling_offsets.csv")
coeff_df.to_csv(coeff_path, index=False)
result_df.to_csv(result_path, index=False)

print(f"\nCoefficients: {coeff_path}")
print(f"Offsets:      {result_path}")
print("\nRolling offset summary:")
print(result_df.to_string(index=False))
print("\nTask 06 complete.")
