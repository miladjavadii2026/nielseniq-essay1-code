"""
stage11_pooled_interaction.py
==============================
Pooled category × promotion-depth interaction model.

PURPOSE
-------
Formally tests whether the promotional lift differs statistically across
cereal, snacks, and coffee after controlling for the same three-way fixed
effects. This directly answers Kevin's question: "why do the three
categories respond differently?" — moving from description to inference.

ESTIMATING EQUATION
-------------------
ln(1+units)_{ist} = sum_j [ beta_j * D_j_{ist} ]
                  + sum_j [ gamma_{j,snacks} * (D_j * 1[cat=snacks])_{ist} ]
                  + sum_j [ gamma_{j,coffee} * (D_j * 1[cat=coffee])_{ist} ]
                  + alpha_{is} + alpha_{st} + alpha_{it} + epsilon_{ist}

Where:
  - D_j = seven promotion-depth bin indicators (D1=ref omitted)
  - beta_j = depth-bin coefficients for cereal (the baseline category)
  - gamma_{j,snacks} = differential lift for snacks vs cereal at bin j
  - gamma_{j,coffee}  = differential lift for coffee vs cereal at bin j
  - alpha_{is}  = store x UPC fixed effect
  - alpha_{st}  = store x week fixed effect
  - alpha_{it}  = UPC x week fixed effect
  - Standard errors: two-way clustered by store and UPC

OUTPUT
------
Table 11A: Pooled interaction model — beta_j (cereal baseline) + gamma differentials
Table 11B: Joint F-tests for snacks and coffee differentials
Saved to: G:/NielsenIQ_Phase2/06_results/stage11_pooled_interaction/

CONSTRAINTS (non-negotiable)
------------------------------
- All source data under G:/NielsenIQ are READ-ONLY
- All generated files written under G:/NielsenIQ_Phase2
- No new Dominick's estimation — this uses NielsenIQ data only
- Dominick's Phase 1 is locked benchmark (not re-estimated here)
"""

import os
import pandas as pd
import numpy as np
import pyfixest as pf

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_RMS      = r"G:\NielsenIQ"                                  # READ-ONLY
BASE_OUT      = r"G:\NielsenIQ_Phase2\06_results\stage11_pooled_interaction"
STAGE2_DIR    = r"G:\NielsenIQ_Phase2\02_analysis_variables"

os.makedirs(BASE_OUT, exist_ok=True)

# ── Category configuration ─────────────────────────────────────────────────
# Same module codes locked in Stage 1 manifest
CATEGORIES = {
    "cereal": {"module": 1344, "years": range(2010, 2021), "exclude": [2012]},
    "snacks": {"module": 1323, "years": range(2011, 2021), "exclude": []},
    "coffee": {"module": 1463, "years": range(2011, 2021), "exclude": []},
}

# ── Promotion-depth bin definitions ───────────────────────────────────────
# Must match Stage 2 exactly (locked)
BIN_COLS  = ["D1_4b", "D2_4b", "D3_4b", "D4_4b",
             "D5_4b", "D6_4b", "D7_4b"]
BIN_LABELS = {
    "D1_4b": "0–5%",
    "D2_4b": "5–10%",
    "D3_4b": "10–20%",
    "D4_4b": "20–30%",
    "D5_4b": "30–40%",
    "D6_4b": "40–50%",
    "D7_4b": "50%+",
}
REF_BIN = "D1_4b"  # omitted reference (0–5%)


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Load and pool analysis panels
# ══════════════════════════════════════════════════════════════════════════

def load_category(cat_name: str, cfg: dict) -> pd.DataFrame:
    """
    Load Stage 2 parquet files for all years of a category,
    concatenate, and add a category indicator column.
    """
    frames = []
    module = cfg["module"]
    exclude = set(cfg.get("exclude", []))

    for year in cfg["years"]:
        if year in exclude:
            print(f"  Skipping {cat_name} {year} (excluded)")
            continue

        fpath = os.path.join(
            STAGE2_DIR, cat_name,
            f"analysis_panel_{cat_name}_{year}.parquet"
        )
        if not os.path.exists(fpath):
            print(f"  WARNING: not found — {fpath}")
            continue

        df = pd.read_parquet(fpath)
        df["category"] = cat_name
        frames.append(df)
        print(f"  Loaded {cat_name} {year}: {len(df):,} rows")

    if not frames:
        raise FileNotFoundError(f"No data found for {cat_name}")

    return pd.concat(frames, ignore_index=True)


print("=" * 60)
print("Stage 11: Pooled Category Interaction Model")
print("=" * 60)

cat_frames = []
for cat_name, cfg in CATEGORIES.items():
    print(f"\nLoading {cat_name}...")
    df_cat = load_category(cat_name, cfg)
    cat_frames.append(df_cat)

panel = pd.concat(cat_frames, ignore_index=True)
print(f"\nTotal pooled observations: {len(panel):,}")
print(f"Category counts:\n{panel['category'].value_counts()}")


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Construct interaction terms
# ══════════════════════════════════════════════════════════════════════════

# Ensure bin columns are integer (0/1)
for col in BIN_COLS:
    if col not in panel.columns:
        raise ValueError(f"Missing bin column: {col}. Check Stage 2 output.")
    panel[col] = panel[col].fillna(0).astype(int)

# Category dummies (cereal = baseline)
panel["is_snacks"] = (panel["category"] == "snacks").astype(int)
panel["is_coffee"] = (panel["category"] == "coffee").astype(int)

# Interaction terms: D_j × is_snacks, D_j × is_coffee
# Skip D1_4b (reference bin)
interaction_bins = [b for b in BIN_COLS if b != REF_BIN]

for bin_col in interaction_bins:
    panel[f"{bin_col}_snacks"] = panel[bin_col] * panel["is_snacks"]
    panel[f"{bin_col}_coffee"] = panel[bin_col] * panel["is_coffee"]

print(f"\nInteraction terms created: {len(interaction_bins) * 2} columns")

# Verify outcome variable
if "ln_units" not in panel.columns:
    if "units" in panel.columns:
        panel["ln_units"] = np.log1p(panel["units"])
        print("Computed ln_units = log(1 + units)")
    else:
        raise ValueError("Neither 'ln_units' nor 'units' found in panel.")

# Fixed effects — use same FE triplet as Stages 3–5
# store_upc, store_week, upc_week
for fe_col in ["store_upc", "store_week", "upc_week"]:
    if fe_col not in panel.columns:
        # Reconstruct if not present
        if fe_col == "store_upc":
            panel["store_upc"] = panel["store_code"].astype(str) + "_" + panel["upc"].astype(str)
        elif fe_col == "store_week":
            panel["store_week"] = panel["store_code"].astype(str) + "_" + panel["week_end"].astype(str)
        elif fe_col == "upc_week":
            panel["upc_week"] = panel["upc"].astype(str) + "_" + panel["week_end"].astype(str)
        print(f"Reconstructed FE column: {fe_col}")


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — Estimate pooled interaction model
# ══════════════════════════════════════════════════════════════════════════

# Build RHS: depth bins (excl. ref) + interactions
depth_rhs   = " + ".join(interaction_bins)
int_snacks  = " + ".join([f"{b}_snacks" for b in interaction_bins])
int_coffee  = " + ".join([f"{b}_coffee" for b in interaction_bins])

formula = (
    f"ln_units ~ {depth_rhs} + {int_snacks} + {int_coffee}"
    f" | store_upc + store_week + upc_week"
)

print(f"\nEstimating: {formula[:120]}...")
print("This may take several minutes on the full panel.\n")

fit = pf.feols(formula, data=panel, vcov={"CRV1": "store_code + upc"})
print(fit.summary())


# ══════════════════════════════════════════════════════════════════════════
# Step 4 — Build output tables
# ══════════════════════════════════════════════════════════════════════════

coef = fit.coef()
se   = fit.se()
pval = fit.pvalue()
tstat = fit.tstat()

def sig_stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ── Table 11A: Main coefficients ──────────────────────────────────────────
rows_A = []
for b in interaction_bins:
    label = BIN_LABELS.get(b, b)
    b_c  = coef.get(b, np.nan)
    b_se = se.get(b, np.nan)
    b_p  = pval.get(b, np.nan)

    sn_k = f"{b}_snacks"
    sn_c = coef.get(sn_k, np.nan)
    sn_se= se.get(sn_k, np.nan)
    sn_p = pval.get(sn_k, np.nan)

    cf_k = f"{b}_coffee"
    cf_c = coef.get(cf_k, np.nan)
    cf_se= se.get(cf_k, np.nan)
    cf_p = pval.get(cf_k, np.nan)

    rows_A.append({
        "Bin": label,
        "Cereal (baseline)": f"{b_c:.4f}{sig_stars(b_p)}",
        "Cereal SE": f"({b_se:.4f})",
        "Snacks diff.": f"{sn_c:+.4f}{sig_stars(sn_p)}" if not np.isnan(sn_c) else "—",
        "Snacks SE": f"({sn_se:.4f})" if not np.isnan(sn_se) else "",
        "Coffee diff.": f"{cf_c:+.4f}{sig_stars(cf_p)}" if not np.isnan(cf_c) else "—",
        "Coffee SE": f"({cf_se:.4f})" if not np.isnan(cf_se) else "",
    })

tbl_A = pd.DataFrame(rows_A)

print("\nTable 11A: Pooled interaction model")
print(tbl_A.to_string(index=False))

tbl_A_path = os.path.join(BASE_OUT, "table11A_pooled_interaction.csv")
tbl_A.to_csv(tbl_A_path, index=False)
print(f"\nSaved: {tbl_A_path}")


# ── Table 11B: Joint F-tests ──────────────────────────────────────────────
print("\nJoint F-tests:")

# Test: all snacks interactions = 0
snacks_vars = [f"{b}_snacks" for b in interaction_bins]
coffee_vars = [f"{b}_coffee" for b in interaction_bins]

rows_B = []
for group_label, var_list in [
    ("All snacks interactions = 0", snacks_vars),
    ("All coffee interactions = 0", coffee_vars),
    ("All interactions = 0 (joint)", snacks_vars + coffee_vars),
]:
    # Build restriction matrix for Wald test
    try:
        # pyfixest wald_test (available in recent versions)
        wald = fit.wald_test(rnames=var_list)
        f_stat = wald["statistic"].values[0]
        f_pval = wald["pvalue"].values[0]
        rows_B.append({
            "Hypothesis": group_label,
            "F-statistic": f"{f_stat:.3f}",
            "p-value": f"{f_pval:.4f}",
            "Result": "Reject H0" if f_pval < 0.05 else "Fail to reject"
        })
        print(f"  {group_label}: F={f_stat:.3f}, p={f_pval:.4f}")
    except Exception as e:
        # Fallback: report that joint test should be run manually
        rows_B.append({
            "Hypothesis": group_label,
            "F-statistic": "see note",
            "p-value": "see note",
            "Result": f"Manual test required: {str(e)[:60]}"
        })
        print(f"  WARNING: Wald test failed for '{group_label}': {e}")

tbl_B = pd.DataFrame(rows_B)
tbl_B_path = os.path.join(BASE_OUT, "table11B_joint_tests.csv")
tbl_B.to_csv(tbl_B_path, index=False)
print(f"\nTable 11B saved: {tbl_B_path}")

print(tbl_B.to_string(index=False))


# ── Summary note ──────────────────────────────────────────────────────────
note = """
Notes: Table 11A reports the depth-bin coefficients for cereal (the baseline
category) and the differential coefficients for snacks and coffee relative to
cereal. A positive snacks/coffee differential at bin j means that category
responds more strongly to a j-bin discount than cereal does. A negative
differential means a weaker response. The reference bin is D1 (0-5% discount).
Standard errors are two-way clustered by store and UPC.
*** p<0.01, ** p<0.05, * p<0.10.

Table 11B reports Wald tests of the joint hypothesis that all bin differentials
for a given category are simultaneously zero. Rejection of this hypothesis
confirms that the cross-category differences documented in Table 2 (individual
category models) are statistically significant in the pooled estimation.

Source: Stage 2 analysis panels, all years. NielsenIQ RMS data (READ-ONLY).
Output: G:/NielsenIQ_Phase2/06_results/stage11_pooled_interaction/
"""

note_path = os.path.join(BASE_OUT, "table11_notes.txt")
with open(note_path, "w") as f:
    f.write(note)
print(f"\nNotes saved: {note_path}")
print("\nStage 11 complete.")
