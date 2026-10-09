"""
task07_promo_duration.py
=========================
PURPOSE
-------
Separate single-week and multi-week promotions to test whether promotion
duration affects (a) the magnitude of promotional lift and (b) the post-
promotion offset ratio.

ECONOMIC MOTIVATION
-------------------
A one-week promotion creates a single-window acceleration opportunity.
A three-week promotion gives households more time to respond, potentially:
  - Larger total purchase acceleration (more households enter the window)
  - Longer post-promotion payback period
  - OR: diminishing returns if households stock up in week 1 and stop buying
    in weeks 2-3 (pantry saturation)

Empirically, if multi-week promotions have LOWER per-week lift but HIGHER
cumulative offset, that would be consistent with early stockpiling and
pantry saturation. If they have HIGHER per-week lift and similar offset,
that suggests the extended window recruits additional households.

DESIGN
------
1. Identify promotion spell duration: count consecutive promo_event==1 weeks
   for each store-UPC.
2. Create duration bins: D_1wk (duration=1), D_2_3wk (duration=2-3),
   D_4plus (duration>=4).
3. For each category, estimate:
   a. FE model of log_units with duration × depth_bin interactions:
      allows same-week lift to differ by promotion duration
   b. Event study stratified by duration group:
      compares offset ratios across duration groups
4. Output: lift by duration, offset by duration.

NOTE ON IDENTIFICATION
-----------------------
Duration is potentially endogenous (stores may run longer promotions on
high-demand products). The store-UPC FE absorbs permanent product-level
differences. Residual endogeneity is discussed in the output notes.
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyfixest as pf

# ── Config ─────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task07_promo_duration"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES  = ["cereal", "coffee", "snacks"]
SAMPLE_FRAC = 0.005
POST_WINDOW = 8
PRE_WINDOW  = 4
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ── Helper: load and annotate promotion spells ─────────────────────────────

def load_with_spell_duration(cat: str, frac: float) -> pd.DataFrame:
    fpath = os.path.join(STAGE2, f"rms_{cat}_analysis.parquet")
    pf_file = pq.ParquetFile(fpath)

    stores = set()
    for batch in pf_file.iter_batches(batch_size=100_000, columns=["store_code_uc"]):
        stores.update(batch.column("store_code_uc").to_pylist())
    sampled = set(rng.choice(sorted(stores),
                             size=max(1, int(len(stores) * frac)),
                             replace=False))

    batches = []
    needed_cols = ["store_code_uc", "upc", "week_end", "promo_event", "depth_bin", "log_units", "units"]
    available = set(pf_file.schema.names)
    needed_cols = [c for c in needed_cols if c in available]
    for batch in pf_file.iter_batches(batch_size=200_000, columns=needed_cols):
        pdf = batch.to_pandas()
        batches.append(pdf[pdf["store_code_uc"].isin(sampled)])
    panel = pd.concat(batches, ignore_index=True)
    panel = panel.sort_values(["store_code_uc", "upc", "week_end"])

    # ── Spell identification ──────────────────────────────────────────────
    # Mark start of each promotion spell
    panel["promo_lag"] = (panel.groupby(["store_code_uc", "upc"])["promo_event"]
                          .shift(1).fillna(0))
    panel["spell_start"] = ((panel["promo_event"] == 1) &
                            (panel["promo_lag"] == 0)).astype(int)
    panel["spell_id"] = (panel.groupby(["store_code_uc", "upc"])["spell_start"]
                         .cumsum())
    panel.loc[panel["promo_event"] == 0, "spell_id"] = np.nan

    # Compute spell duration
    spell_dur = (panel[panel["promo_event"] == 1]
                 .groupby(["store_code_uc", "upc", "spell_id"])
                 .size()
                 .reset_index(name="spell_duration"))
    panel = panel.merge(spell_dur, on=["store_code_uc", "upc", "spell_id"], how="left")
    panel["spell_duration"] = panel["spell_duration"].fillna(0).astype(int)

    # Duration bins
    panel["dur_bin"] = pd.cut(
        panel["spell_duration"],
        bins=[-1, 0, 1, 3, 999],
        labels=["not_promo", "1wk", "2_3wk", "4plus_wk"]
    )

    if "log_units" not in panel.columns:
        panel["log_units"] = np.log1p(panel["units"])

    panel["store_upc"]  = panel["store_code_uc"].astype(str) + "_" + panel["upc"].astype(str)
    panel["store_week"] = panel["store_code_uc"].astype(str) + "_" + panel["week_end"].astype(str)
    panel["upc_week"]   = panel["upc"].astype(str) + "_" + panel["week_end"].astype(str)

    return panel


# ── Main ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("Task 07: Promotion Duration Analysis")
print("=" * 65)

all_lift = []
all_offset = []

for cat in CATEGORIES:
    print(f"\n{cat.upper()}")
    panel = load_with_spell_duration(cat, SAMPLE_FRAC)
    print(f"  Panel rows: {len(panel):,}")

    # Spell duration distribution
    dur_dist = panel[panel["spell_duration"] > 0]["spell_duration"].value_counts().sort_index()
    print("  Spell duration distribution:")
    for d, n in dur_dist.items():
        print(f"    {d} weeks: {n:,} obs")

    # ── Part A: lift by duration bin ─────────────────────────────────────
    # Create depth dummies from depth_bin
    bins = ["D2", "D3", "D4", "D5", "D6", "D7"]
    for b in bins:
        panel[b] = (panel["depth_bin"] == b).astype(np.int8)

    # Interactions: depth_bin × dur_bin (for promoted observations)
    # Simplified: estimate separate FE model for each duration group
    # Then compare lift at D5+ depth bins

    lift_rows = []
    for dur_grp, dur_label in [("1wk", "1 week"), ("2_3wk", "2-3 weeks"), ("4plus_wk", "4+ weeks")]:
        sub = panel[(panel["dur_bin"] == dur_grp) |
                    (panel["promo_event"] == 0)].copy()
        # Only keep obs where the store-UPC has at least some spells of this duration
        # (to ensure valid store-UPC FE)
        valid_store_upc = sub[sub["dur_bin"] == dur_grp]["store_upc"].unique()
        sub = sub[sub["store_upc"].isin(valid_store_upc)]

        if sub["promo_event"].sum() < 500:
            print(f"  Too few promoted obs for {dur_grp}, skipping lift model")
            continue

        rhs = " + ".join(bins)
        try:
            fit = pf.feols(f"log_units ~ {rhs} | store_upc + store_week + upc_week",
                           data=sub, vcov={"CRV1": "store_code_uc"})
            coefs = dict(fit.coef())
            se_   = dict(fit.se())
            for b in bins:
                lift_rows.append({
                    "category": cat, "duration_group": dur_grp, "dur_label": dur_label,
                    "bin": b, "coef": coefs.get(b, np.nan), "se": se_.get(b, np.nan),
                    "n_obs": len(sub),
                })
            print(f"  {dur_label}: D7 lift = {coefs.get('D7', np.nan):.4f}")
        except Exception as e:
            print(f"  ERROR ({dur_grp}): {e}")

    all_lift.extend(lift_rows)

    # ── Part B: offset ratio by duration group ────────────────────────────
    panel["promo_end"] = ((panel["promo_event"] == 1) &
        (panel.groupby(["store_code_uc", "upc"])["promo_event"].shift(-1).fillna(0) == 0))
    panel["t_end_week"] = np.where(panel["promo_end"], panel["week_end"], np.nan)
    panel["t_end_week"] = panel.groupby(["store_code_uc", "upc"])["t_end_week"].fillna(method="bfill")
    panel["t_end_dur"]  = np.where(panel["promo_end"], panel["dur_bin"].astype(str), np.nan)
    panel["t_end_dur"]  = panel.groupby(["store_code_uc", "upc"])["t_end_dur"].fillna(method="bfill")

    for col in ["week_end", "t_end_week"]:
        dt = pd.to_datetime(panel[col])
        panel[f"yw_{col}"] = dt.dt.year * 53 + dt.dt.isocalendar().week.astype(int)
    panel["event_time"] = panel["yw_week_end"] - panel["yw_t_end_week"]

    taus = [t for t in range(-PRE_WINDOW, POST_WINDOW + 1) if t != -1]
    for tau in taus:
        col = f"tau_{tau:+d}".replace("+", "p").replace("-", "m")
        panel[col] = (panel["event_time"] == tau).astype(np.int8)

    tau_names = [f"tau_{t:+d}".replace("+", "p").replace("-", "m") for t in taus]
    rhs_ev = " + ".join(tau_names)

    for dur_grp in ["1wk", "2_3wk", "4plus_wk"]:
        evpanel = panel[(panel["event_time"] >= -PRE_WINDOW) &
                        (panel["event_time"] <= POST_WINDOW) &
                        (panel["t_end_week"].notna()) &
                        (panel["t_end_dur"] == dur_grp)].copy()
        if len(evpanel) < 500:
            continue
        try:
            fit_ev = pf.feols(f"log_units ~ {rhs_ev} | store_upc + store_week + upc_week",
                              data=evpanel, vcov={"CRV1": "store_code_uc"})
            coefs = dict(fit_ev.coef())
            during = sum(coefs.get(f"tau_m{abs(t)}", 0.0) for t in range(-PRE_WINDOW, 0)) + coefs.get("tau_p0", 0.0)
            post   = sum(coefs.get(f"tau_p{t}", 0.0) for t in range(1, POST_WINDOW + 1))
            offset = abs(post) / during if during > 1e-8 else np.nan
            all_offset.append({"category": cat, "duration_group": dur_grp,
                               "offset_ratio": round(offset, 4), "n_events": len(evpanel)})
            print(f"  Offset ({dur_grp}): {offset:.4f}  [{len(evpanel):,} obs]")
        except Exception as e:
            print(f"  ERROR offset ({dur_grp}): {e}")

lift_df   = pd.DataFrame(all_lift)
offset_df = pd.DataFrame(all_offset)

lift_df.to_csv(os.path.join(OUT_DIR, "task07_lift_by_duration.csv"), index=False)
offset_df.to_csv(os.path.join(OUT_DIR, "task07_offset_by_duration.csv"), index=False)

print(f"\nOutputs in {OUT_DIR}")
if not offset_df.empty:
    print("\nOffset ratio by duration:")
    print(offset_df.to_string(index=False))
print("\nTask 07 complete.")
