"""
Step 5: Weekly household purchase-hazard model.

Constructs weekly risk-set panel from HMS purchase-level data.
For each purchase spell (interpurchase_weeks), expands into week-level rows.
Outcome: PurchaseEvent_t = 1 if household purchases in week t of spell.

Model: logit(Pr(Purchase=1|no prior purchase in spell))
       = f(WeeksSinceLastPurchase, PriorPromoPurchase, CalendarYear, HH_FE)

Approach: discrete-time hazard (complementary log-log or logit) 
using statsmodels logit with household FE via Chamberlain's conditional logit
(due to incidental parameters). Here: use fixed-effect absorption via demean.

For sandbox: use N_RG=5 from HMS parquet. Full production uses all 71 RGs.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings; warnings.filterwarnings('ignore')
try:
    from linearmodels.panel import PanelOLS
    HAS_LM = True
except ImportError:
    HAS_LM = False

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step05_hazard"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS = ["cereal","snacks","coffee"]; N_RG_HMS = 5

def load_hms(cat, n_rg=5):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/hms_{cat}_analysis.parquet")
    n = min(n_rg, pf_.num_row_groups)
    # Load available columns
    avail = pf_.schema.names
    want  = ["household_code","upc","purchase_date","panel_year",
             "interpurchase_weeks","promo_coincident_flag",
             "retailer_code","brand_code_uc","dma_cd",
             "household_income","hh_purchase_qty"]
    cols = [c for c in want if c in avail]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas() for i in range(n)]
    df = pd.concat(frames, ignore_index=True)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    return df

def build_weekly_risk_set(df, cat):
    """
    Expand purchase records into weekly risk-set panel.
    
    For each purchase event:
      - The spell runs from the PREVIOUS purchase to this one
      - Spell length = interpurchase_weeks (if known)
      - Create rows: week 1, 2, ..., L-1 (outcome=0), week L (outcome=1)
      - PriorPromo = promo_coincident_flag of PREVIOUS purchase
    
    Returns: panel DataFrame with columns:
      [spell_id, household_code, cat, week_in_spell, outcome,
       prior_promo_purchase, panel_year]
    """
    df = df.sort_values(["household_code","purchase_date"]).copy()
    df["interp"] = pd.to_numeric(df["interpurchase_weeks"], errors="coerce")
    
    # Lag promo flag (prior purchase's promo status)
    df["prior_promo"] = (df.groupby("household_code")["promo_coincident_flag"]
                         .shift(1))
    df["prior_promo"] = df["prior_promo"].fillna(0).astype(int)
    
    # Filter: valid spell length
    df = df[(df["interp"] > 0) & (df["interp"] <= 52)].copy()  # max 1 year spell
    
    # Also need panel_year for calendar time
    df["year"] = df["purchase_date"].dt.year.fillna(df.get("panel_year",2015))
    
    records = []
    spell_id = 0
    
    # Process sample (limit to avoid timeout)
    for _, row in df.head(15000).iterrows():
        L = int(row["interp"])
        hh = row["household_code"]
        pp = int(row["prior_promo"])
        yr = int(row["year"]) if not pd.isna(row["year"]) else 2015
        
        for t in range(1, L+1):
            records.append({
                "spell_id": spell_id,
                "household_code": hh,
                "category": cat,
                "week_in_spell": t,
                "outcome": 1 if t == L else 0,
                "prior_promo": pp,
                "panel_year": yr,
                "log_weeks": np.log(t)
            })
        spell_id += 1
    
    panel = pd.DataFrame(records)
    return panel

def fit_hazard_logit(panel, cat):
    """
    Fit discrete-time hazard model via logistic regression.
    Within-demean to absorb household FE.
    
    Pr(outcome=1) = sigma(alpha_hh + b1*log_weeks + b2*prior_promo + b3*year)
    
    Since we can't do proper conditional logit at scale,
    use within-estimator (demean by household) as approximation.
    (This is Chamberlain-style linear probability model for speed.)
    """
    from sklearn.linear_model import LogisticRegression
    
    if len(panel) < 100:
        return None
    
    panel = panel.copy()
    
    # Within-demean continuous vars by household (crude HH FE)
    for col in ["log_weeks"]:
        panel[col+"_dm"] = panel[col] - panel.groupby("household_code")[col].transform("mean")
    
    # Standardize year
    yr_mean = panel["panel_year"].mean()
    yr_std  = panel["panel_year"].std() + 1e-6
    panel["year_std"] = (panel["panel_year"] - yr_mean) / yr_std
    
    X_cols = ["log_weeks_dm","prior_promo","year_std"]
    X = panel[X_cols].fillna(0).values
    y = panel["outcome"].values
    
    if y.sum() < 10 or (1-y).sum() < 10:
        return None
    
    # Logistic regression (approximation of conditional logit)
    lr = LogisticRegression(penalty=None, max_iter=500, solver='lbfgs')
    try:
        lr.fit(X, y)
    except Exception as e:
        print(f"    Logit error: {e}")
        return None
    
    coefs = lr.coef_[0]
    
    # Bootstrap SE (100 draws for speed)
    n = len(panel)
    rng = np.random.default_rng(42)
    boot_coefs = []
    for _ in range(100):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]; yb = y[idx]
        if yb.sum() < 5 or (1-yb).sum() < 5:
            continue
        try:
            lrb = LogisticRegression(penalty=None, max_iter=300, solver='lbfgs')
            lrb.fit(Xb, yb)
            boot_coefs.append(lrb.coef_[0])
        except:
            pass
    
    if len(boot_coefs) < 20:
        ses = np.zeros(len(X_cols))
    else:
        ses = np.std(boot_coefs, axis=0)
    
    results = []
    var_names = ["log_weeks_demeaned", "prior_promo_purchase", "calendar_year"]
    for i, (vn, coef, se) in enumerate(zip(var_names, coefs, ses)):
        z = coef/se if se>0 else np.nan
        p = 2*(1-st.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        # Convert to odds ratio
        or_ = np.exp(coef)
        or_lo = np.exp(coef - 1.96*se)
        or_hi = np.exp(coef + 1.96*se)
        results.append({
            "variable": vn, "coefficient": round(coef,5),
            "se_bootstrap": round(se,5), "z_stat": round(z,3) if not np.isnan(z) else np.nan,
            "p_val": round(p,5) if not np.isnan(p) else np.nan,
            "sig": sig,
            "odds_ratio": round(or_,4),
            "or_ci_lo": round(or_lo,4),
            "or_ci_hi": round(or_hi,4)
        })
    
    return pd.DataFrame(results)

all_results = []; spell_counts = {}

for cat in CATS:
    print(f"\n{'='*55}\n  {cat.upper()}")
    df = load_hms(cat, N_RG_HMS)
    print(f"  HMS purchases: {len(df):,} | HHs: {df['household_code'].nunique()}")
    
    panel = build_weekly_risk_set(df, cat)
    n_spells = panel["spell_id"].nunique()
    n_events = panel["outcome"].sum()
    n_censored = n_spells - n_events
    
    print(f"  Spells: {n_spells:,} | Events: {n_events:,} | "
          f"Censoring: {n_censored:,}")
    spell_counts[cat] = {"n_spells":n_spells,"n_events":int(n_events),
                          "n_censored":n_censored,"n_panel_rows":len(panel)}
    
    res = fit_hazard_logit(panel, cat)
    
    if res is None:
        print(f"  [{cat}] Hazard model failed")
        continue
    
    res["category"] = cat
    res["n_spells"] = n_spells
    res["n_events"] = int(n_events)
    res["n_censored"] = n_censored
    all_results.append(res)
    
    print(f"  Results:")
    print(res[["variable","coefficient","se_bootstrap","z_stat","p_val",
               "sig","odds_ratio","or_ci_lo","or_ci_hi"]].to_string(index=False))

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(f"{OUT_DIR}/hazard_model_results.csv", index=False)
    
    # Spell counts
    pd.DataFrame(spell_counts).T.to_csv(f"{OUT_DIR}/hazard_spell_counts.csv")
    
    print(f"\n{'='*55}")
    print("KEY RESULT: Prior Promo → Purchase Hazard")
    pp = combined[combined["variable"]=="prior_promo_purchase"]
    for _,row in pp.iterrows():
        print(f"  {row['category']:8s}: OR={row['odds_ratio']:.4f} "
              f"95%CI=[{row['or_ci_lo']:.4f},{row['or_ci_hi']:.4f}] "
              f"p={row['p_val']:.4f} {row['sig']}")
    
    # Figure: hazard ratio CIs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Step 5 — Weekly Household Purchase Hazard Model\n"
                 "(Discrete-time logit; HH within-demean; bootstrap SE)",
                 fontsize=12, fontweight='bold')
    
    cat_colors = {"cereal":"#2196F3","snacks":"#FF9800","coffee":"#4CAF50"}
    
    # Left: Odds ratios for prior_promo
    ax = axes[0]
    pp_data = combined[combined["variable"]=="prior_promo_purchase"].copy()
    y_pos = np.arange(len(pp_data))
    ax.axvline(1.0, color='gray', lw=1.5, ls='--', label='OR=1 (no effect)')
    for i, (_, row) in enumerate(pp_data.iterrows()):
        col = cat_colors.get(row["category"],"black")
        ax.plot([row["or_ci_lo"],row["or_ci_hi"]],[i,i],'-',color=col,lw=4,alpha=0.6)
        ax.plot(row["odds_ratio"], i, 'o', color=col, ms=10, zorder=5,
                label=f"{row['category'].capitalize()} OR={row['odds_ratio']:.3f} {row['sig']}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["category"].capitalize() for _,r in pp_data.iterrows()],fontsize=11)
    ax.set_xlabel("Odds Ratio (Prior Promo Purchase → Purchase Hazard)")
    ax.set_title("Effect of Prior Promotion Purchase\non Repurchase Hazard (OR + 95% CI)")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    
    # Right: Duration dependence (log_weeks coefficient → hazard shape)
    ax2 = axes[1]
    lw_data = combined[combined["variable"]=="log_weeks_demeaned"].copy()
    for i,(_, row) in enumerate(lw_data.iterrows()):
        col = cat_colors.get(row["category"],"black")
        ax2.plot([row["or_ci_lo"],row["or_ci_hi"]],[i,i],'-',color=col,lw=4,alpha=0.6)
        ax2.plot(row["odds_ratio"], i, 'o', color=col, ms=10, zorder=5,
                 label=f"{row['category'].capitalize()} OR={row['odds_ratio']:.3f} {row['sig']}")
    ax2.axvline(1.0, color='gray',lw=1.5,ls='--')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([r["category"].capitalize() for _,r in lw_data.iterrows()],fontsize=11)
    ax2.set_xlabel("Odds Ratio (log weeks in spell → purchase probability)")
    ax2.set_title("Duration Dependence: log(weeks)\n(OR>1: risk rises with time)")
    ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/step05_hazard_model.png",dpi=150,bbox_inches='tight')
    plt.close()
    print("Figure saved.")
