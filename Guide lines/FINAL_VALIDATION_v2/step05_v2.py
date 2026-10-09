"""Step 5 v2: Hazard model — fast version."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step05_hazard"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS = ["cereal","snacks","coffee"]

def load_hms(cat, n_rg=2):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/hms_{cat}_analysis.parquet")
    cols = ["household_code","purchase_date","panel_year",
            "interpurchase_weeks","promo_coincident_flag"]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas()
              for i in range(min(n_rg, pf_.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    return df

def expand_spells(df, max_rows=8000):
    """Vectorized spell expansion — much faster than row-by-row."""
    df = df.copy()
    df["interp"] = pd.to_numeric(df["interpurchase_weeks"], errors="coerce")
    # Lag promo
    df = df.sort_values(["household_code","purchase_date"]).reset_index(drop=True)
    df["prior_promo"] = (df.groupby("household_code")["promo_coincident_flag"]
                         .shift(1).fillna(0).astype(int))
    df["year"] = df["purchase_date"].dt.year.fillna(2015)
    df = df[(df["interp"]>=1)&(df["interp"]<=52)].copy()
    df["spell_id"] = np.arange(len(df))
    
    # Repeat each row by interp weeks
    df_rep = df.loc[df.index.repeat(df["interp"].astype(int))].copy().reset_index(drop=True)
    df_rep["week_in_spell"] = df_rep.groupby("spell_id").cumcount() + 1
    
    # Outcome=1 only in final week
    interp_lookup = df.set_index("spell_id")["interp"]
    df_rep["spell_len"] = df_rep["spell_id"].map(interp_lookup).astype(int)
    df_rep["outcome"] = (df_rep["week_in_spell"] == df_rep["spell_len"]).astype(int)
    df_rep["log_weeks"] = np.log(df_rep["week_in_spell"] + 1e-6)
    
    # Limit size
    if len(df_rep) > max_rows:
        df_rep = df_rep.sample(max_rows, random_state=42).reset_index(drop=True)
    
    return df_rep, df["spell_id"].nunique()

def logit_hazard(panel):
    """Fit logit hazard model with HH within-demean."""
    # Within-demean log_weeks by household
    panel = panel.copy()
    panel["lw_dm"] = (panel["log_weeks"] - 
                      panel.groupby("household_code")["log_weeks"].transform("mean"))
    yr_m = panel["year"].mean(); yr_s = panel["year"].std()+1e-6
    panel["yr_std"] = (panel["year"]-yr_m)/yr_s
    
    X = panel[["lw_dm","prior_promo","yr_std"]].fillna(0).values
    y = panel["outcome"].values
    
    if y.sum() < 10: return None
    
    lr = LogisticRegression(penalty=None, max_iter=300, solver='lbfgs')
    lr.fit(X, y)
    coefs = lr.coef_[0]
    
    # Sandwich SE via 50-draw bootstrap
    n = len(panel); rng = np.random.default_rng(42)
    boots = []
    for _ in range(50):
        idx = rng.integers(0, n, size=n)
        Xb,yb = X[idx],y[idx]
        if yb.sum()<5 or (1-yb).sum()<5: continue
        try:
            lr2 = LogisticRegression(penalty=None,max_iter=200,solver='lbfgs')
            lr2.fit(Xb,yb); boots.append(lr2.coef_[0])
        except: pass
    ses = np.std(boots,axis=0) if len(boots)>=10 else np.zeros(3)
    
    vars_ = ["log_weeks (duration dep.)", "prior_promo_purchase", "calendar_year"]
    rows = []
    for vn,c,s in zip(vars_,coefs,ses):
        z = c/s if s>0 else np.nan
        p = 2*(1-st.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        sig = "***" if (p is not np.nan and p<0.001) else ("**" if (p is not np.nan and p<0.01) else "*" if (p is not np.nan and p<0.05) else "")
        rows.append({"variable":vn,"coef":round(c,5),"se":round(s,5),
                     "OR":round(np.exp(c),4),
                     "OR_lo":round(np.exp(c-1.96*s),4),
                     "OR_hi":round(np.exp(c+1.96*s),4),
                     "z":round(z,3) if not np.isnan(z) else np.nan,
                     "p":round(p,5) if not np.isnan(p) else np.nan,"sig":sig})
    return pd.DataFrame(rows)

all_res = []
spell_info = {}

for cat in CATS:
    print(f"\n{'='*45}\n  {cat.upper()}")
    df = load_hms(cat, 2)
    print(f"  HMS rows: {len(df):,} HHs: {df['household_code'].nunique()}")
    
    panel, n_sp = expand_spells(df)
    n_ev = panel["outcome"].sum()
    print(f"  Panel rows: {len(panel):,} | Spells: {n_sp:,} | Events: {n_ev:,}")
    
    spell_info[cat] = {"n_spells":n_sp,"n_events":int(n_ev),
                       "n_panel_rows":len(panel),
                       "censoring_rule":"interp_weeks<=52; end-of-observation censored at final wave"}
    
    res = logit_hazard(panel)
    if res is None:
        print("  Failed"); continue
    res["category"] = cat
    res["n_spells"]  = n_sp
    res["n_events"]  = int(n_ev)
    all_res.append(res)
    print(res[["variable","coef","se","OR","OR_lo","OR_hi","p","sig"]].to_string(index=False))

if all_res:
    out = pd.concat(all_res, ignore_index=True)
    out.to_csv(f"{OUT_DIR}/hazard_model_results.csv", index=False)
    pd.DataFrame(spell_info).T.to_csv(f"{OUT_DIR}/hazard_spell_counts.csv")
    
    # Figure
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Step 5 — Weekly Household Purchase-Hazard Model\n"
                 "(Discrete-time logit; within-demean HH FE; bootstrap SE; N_RG=2)",
                 fontsize=11,fontweight='bold')
    
    cc = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
    
    for ax_i,(var_frag,ax_title) in enumerate([
        ("prior_promo","Prior Promo Purchase → Repurchase Hazard\n(OR>1: faster repurchase after promo)"),
        ("log_weeks","Duration Dependence: log(weeks in spell)\n(OR>1: hazard rises over time)")
    ]):
        ax = axes[ax_i]
        sub = out[out["variable"].str.contains(var_frag)].copy()
        yp = np.arange(len(sub))
        ax.axvline(1.0,color='gray',lw=1.5,ls='--',alpha=0.7)
        for i,(_,row) in enumerate(sub.iterrows()):
            col = cc.get(row["category"],"black")
            ax.barh(i,row["OR"]-1,left=1,color=col,alpha=0.7,height=0.5)
            ax.errorbar(row["OR"],i,
                       xerr=[[row["OR"]-row["OR_lo"]],[row["OR_hi"]-row["OR"]]],
                       fmt='o',color=col,ms=8,capsize=5,lw=2,
                       label=f"{row['category'].capitalize()}: OR={row['OR']:.3f} {row['sig']}")
        ax.set_yticks(yp)
        ax.set_yticklabels([r["category"].capitalize() for _,r in sub.iterrows()],fontsize=11)
        ax.set_xlabel("Odds Ratio")
        ax.set_title(ax_title)
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/step05_hazard_model.png",dpi=150,bbox_inches='tight')
    plt.close()
    print("\nFigure saved.")
    
    # Key result
    pp = out[out["variable"].str.contains("prior_promo")]
    print(f"\nKEY: Prior-Promo → Hazard")
    for _,r in pp.iterrows():
        print(f"  {r['category']:8s}: OR={r['OR']:.4f} CI=[{r['OR_lo']:.4f},{r['OR_hi']:.4f}] {r['sig']}")
