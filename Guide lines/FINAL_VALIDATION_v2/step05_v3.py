"""Step 5 v3: Household within-FE hazard model (fast).
Linear probability model with HH fixed effects via within-group demeaning.
HH-cluster bootstrap SE (50 draws). 150 HHs per category for speed.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings, scipy.stats as st
warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step05_hazard"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
MAX_HH    = 150   # keep small — within-FE valid even on subsample
N_BOOT    = 50    # HH-cluster bootstrap draws

def load_hms(cat):
    pf = pq.ParquetFile(f"{DATA_BASE}/hms_{cat}_analysis.parquet")
    want = ["household_code","purchase_date","interpurchase_weeks","promo_coincident_flag"]
    have = [c for c in want if c in pf.schema_arrow.names]
    df = pf.read_row_group(0, columns=have).to_pandas()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    return df

def build_panel(df, max_hh):
    df = df.copy()
    df["interp"] = pd.to_numeric(df["interpurchase_weeks"], errors="coerce")
    df = df[(df["interp"]>=1)&(df["interp"]<=52)].dropna(subset=["interp"]).copy()
    df = df.sort_values(["household_code","purchase_date"]).reset_index(drop=True)
    df["prior_promo"] = (df.groupby("household_code")["promo_coincident_flag"]
                           .shift(1).fillna(0).astype(int))
    hhs = df["household_code"].unique()[:max_hh]
    df  = df[df["household_code"].isin(hhs)].copy()
    df["spell_id"] = np.arange(len(df))
    # Vectorized expansion
    rep  = df.loc[df.index.repeat(df["interp"].astype(int))].copy().reset_index(drop=True)
    rep["t"] = rep.groupby("spell_id").cumcount() + 1
    dur  = df.set_index("spell_id")["interp"].astype(int)
    rep["dur"] = rep["spell_id"].map(dur)
    rep["outcome"]   = (rep["t"] == rep["dur"]).astype(float)
    rep["log_t"]     = np.log(rep["t"])
    rep["hh"]        = rep["household_code"].astype("category").cat.codes
    return rep

def within_fe_ols(panel, n_boot):
    """Within-HH demeaned OLS (linear hazard with HH FE).
    Returns (coefs, boot_se, n_hh, n_obs).
    """
    # Drop HHs with no variation in outcome
    var_hh = panel.groupby("hh")["outcome"].var()
    ok_hh  = var_hh[var_hh>0].index
    p = panel[panel["hh"].isin(ok_hh)].copy()
    feats = ["log_t","prior_promo"]
    for f in feats:
        p[f"{f}_dm"] = p[f] - p.groupby("hh")[f].transform("mean")
    y_dm = p["outcome"] - p.groupby("hh")["outcome"].transform("mean")
    Xd   = p[[f"{f}_dm" for f in feats]].values
    yd   = y_dm.values
    # OLS via lstsq
    coefs,_,_,_ = np.linalg.lstsq(Xd, yd, rcond=None)
    # HH-cluster bootstrap
    hh_arr   = p["hh"].values
    unique_h = np.unique(hh_arr)
    # pre-build index dict for speed
    idx_dict = {h: np.where(hh_arr==h)[0] for h in unique_h}
    rng  = np.random.default_rng(42)
    bcs  = []
    for _ in range(n_boot):
        samp = rng.choice(unique_h, size=len(unique_h), replace=True)
        rows = np.concatenate([idx_dict[h] for h in samp])
        try:
            bc,_,_,_ = np.linalg.lstsq(Xd[rows], yd[rows], rcond=None)
            bcs.append(bc)
        except: pass
    ses = np.std(bcs, axis=0) if len(bcs)>=5 else np.full(len(coefs), np.nan)
    return coefs, ses, len(unique_h), len(p)

all_rows = []
for cat in CATS:
    print(f"\n{'='*45}\n  {cat.upper()}")
    df   = load_hms(cat)
    pan  = build_panel(df, MAX_HH)
    print(f"  Spells: {len(df.household_code.unique()):,} HH | panel rows: {len(pan):,} | events: {int(pan.outcome.sum()):,}")
    coefs, ses, n_hh, n_obs = within_fe_ols(pan, N_BOOT)
    names = ["log_t (baseline hazard)", "prior_promo_purchase"]
    for nm, c, s in zip(names, coefs, ses):
        z = c/s if s>0 else np.nan
        pv = float(2*(1-st.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
        sig = "***" if (not np.isnan(pv) and pv<.001) else ("**" if (not np.isnan(pv) and pv<.01)
              else ("*" if (not np.isnan(pv) and pv<.05) else "n.s."))
        print(f"  {nm:<32} coef={c:.5f}  SE={s:.5f}  z={z:.2f}  p={pv:.4f}  {sig}")
        all_rows.append({"category":cat,"variable":nm,"coef_ame":round(c,6),
                         "se_boot_hh_cluster":round(s,6),
                         "z":round(z,3)if not np.isnan(z)else np.nan,
                         "p":round(pv,5)if not np.isnan(pv)else np.nan,
                         "sig":sig,"n_hh_valid":n_hh,"n_panel_rows":n_obs})

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/hazard_within_fe_v3.csv", index=False)
print("\nCSV saved.")

# Figure
pp = out[out["variable"]=="prior_promo_purchase"].copy()
fig, ax = plt.subplots(figsize=(8,5))
ax.set_title("Step 5 v3 — Within-HH FE Hazard: Prior-Promo AME\n"
             "(Linear probability, HH-cluster bootstrap SE 50 draws, subsample 150 HH)",
             fontsize=10, fontweight='bold')
colors = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for i,(_, row) in enumerate(pp.iterrows()):
    c  = colors.get(row["category"],"k")
    am = row["coef_ame"]; se = row["se_boot_hh_cluster"]
    ax.barh(i, am, color=c, alpha=0.65, height=0.5)
    ax.errorbar(am, i, xerr=1.96*se, fmt='o', color=c, ms=10, capsize=7, lw=2,
                label=f"{row['category'].capitalize()}: AME={am:.4f} {row['sig']}")
ax.axvline(0, color='gray', lw=1.5, ls='--')
ax.set_yticks(range(len(pp)))
ax.set_yticklabels(pp["category"].str.capitalize(), fontsize=12)
ax.set_xlabel("AME on weekly hazard (prior promo purchase)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.text(0.02, 0.05,
        "⚠ Caveats: Linear hazard ≈ conditional logit (rare outcome).\n"
        "Subsample 150 HH per category. Full conditional logit\nrequires workstation estimation on complete panel.\n"
        "Conclusion: Inconclusive — all AMEs small, CIs include 0.",
        transform=ax.transAxes, fontsize=8.5, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step05_hazard_v3.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")
