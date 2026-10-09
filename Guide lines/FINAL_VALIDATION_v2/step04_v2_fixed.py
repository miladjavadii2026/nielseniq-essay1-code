"""Step 4 v2 — Fix: remove unit_price (post-treatment variable) from features."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
import scipy.stats as st

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step04_grouped_dml"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
N_RG      = 1

# FIXED: unit_price REMOVED — it is contemporaneous with the discount treatment
# Features: brand_h, upc_h, week-of-year sinusoid ONLY (pre-treatment / stable)
def load_rms(cat, n_rg=1):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","W5","W10",
            "week_of_year","brand_code_uc"]
    avail = pf_.schema_arrow.names
    cols = [c for c in want if c in avail]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas()
              for i in range(min(n_rg, pf_.num_row_groups))]
    return pd.concat(frames, ignore_index=True)

def build_feats(df):
    df = df.copy()
    df["brand_h"] = df["brand_code_uc"].astype(str).apply(lambda x: hash(x)%200/200)
    df["upc_h"]   = df["upc"].astype(str).apply(lambda x: hash(x)%500/500)
    df["sin_wk"]  = np.sin(2*np.pi*df["week_of_year"].fillna(26)/52)
    df["cos_wk"]  = np.cos(2*np.pi*df["week_of_year"].fillna(26)/52)
    return df

def lz_se(Y_res, W_res, groups):
    G = len(np.unique(groups)); n = len(Y_res)
    theta = float(np.sum(W_res*Y_res)/np.sum(W_res**2))
    eps = Y_res - theta*W_res
    df_ = pd.DataFrame({"s":groups,"Wr":W_res,"ep":eps})
    agg = df_.groupby("s").apply(lambda x: float(np.sum(x["Wr"]*x["ep"]))).values
    meat = np.sum(agg**2)*(G/(G-1))*((n-1)/(n-3))
    se = float(np.sqrt(max(0,meat)/np.sum(W_res**2)**2))
    t_ = theta/se if se>0 else np.nan
    p_ = float(2*(1-st.norm.cdf(abs(t_)))) if not np.isnan(t_) else np.nan
    sig = "***" if p_<0.001 else ("**" if p_<0.01 else ("*" if p_<0.05 else ""))
    return theta, se, t_, p_, sig

LEARNERS = [
    ("Ridge",      Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=1.0))])),
    ("Lasso",      Pipeline([("sc",StandardScaler()),("m",Lasso(alpha=0.005,max_iter=2000))])),
    ("ElasticNet", Pipeline([("sc",StandardScaler()),("m",ElasticNet(alpha=0.005,l1_ratio=0.5,max_iter=2000))])),
]
FEAT_COLS = ["brand_h","upc_h","sin_wk","cos_wk"]

all_rows = []
for cat in CATS:
    df = load_rms(cat, N_RG)
    df = build_feats(df).dropna(subset=["log_units","W5","W10"]).reset_index(drop=True)
    print(f"\n{cat.upper()}: n={len(df):,}  W5_rate={df['W5'].mean():.3f}  W10_rate={df['W10'].mean():.3f}")
    print(f"  Features: {FEAT_COLS}  [unit_price EXCLUDED — post-treatment]")
    
    X = df[FEAT_COLS].fillna(0).values
    stores = df["store_code_uc"].values
    n_stores = len(np.unique(stores))
    
    for w_col in ["W5","W10"]:
        W = df[w_col].astype(float).values
        Y = df["log_units"].values
        
        for lname,learner in LEARNERS:
            gkf = GroupKFold(n_splits=min(5, n_stores))
            Y_res = np.zeros(len(df)); W_res = np.zeros(len(df))
            for tr,te in gkf.split(X, Y, groups=stores):
                learner.fit(X[tr],Y[tr])
                Y_res[te] = Y[te]-learner.predict(X[te])
                learner.fit(X[tr],W[tr])
                W_res[te] = W[te]-learner.predict(X[te])
            
            theta,se,t_,p_,sig = lz_se(Y_res, W_res, stores)
            print(f"  {w_col} {lname:12s}: ATE={theta:.4f}  SE={se:.4f}  t={t_:.2f}  {sig}")
            all_rows.append({"category":cat,"treatment":w_col,"learner":lname,
                             "ate":round(theta,5),"se_clustered":round(se,5),
                             "t_stat":round(t_,3),"p_val":round(p_,6),"sig":sig,
                             "n_obs":len(df),"n_stores":n_stores,
                             "features_note":"brand_h,upc_h,sin_wk,cos_wk — unit_price excluded"})

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/grouped_dml_v2_noprice.csv", index=False)

# Figure
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Step 4 v2 — GroupKFold DML (unit_price REMOVED)\n"
             "Features: brand, UPC, week seasonality only (pre-treatment covariates)",
             fontsize=11,fontweight='bold')
cc = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}

for ax_i,(wt,ax) in enumerate(zip(["W5","W10"],axes)):
    sub = out[out["treatment"]==wt]
    cats_order = ["cereal","snacks","coffee"]
    learners_order = ["Ridge","Lasso","ElasticNet"]
    x_pos = np.arange(len(cats_order))
    w = 0.25
    for j,ln in enumerate(learners_order):
        sub2 = sub[sub["learner"]==ln]
        ates = [sub2[sub2["category"]==c]["ate"].values[0] if len(sub2[sub2["category"]==c])>0 else 0
                for c in cats_order]
        ses  = [sub2[sub2["category"]==c]["se_clustered"].values[0] if len(sub2[sub2["category"]==c])>0 else 0
                for c in cats_order]
        colors = [cc[c] for c in cats_order]
        xp = x_pos + (j-1)*w
        ax.bar(xp, ates, width=w*0.9, label=ln, alpha=0.8,
               color=[f"{cc[c]}" for c in cats_order], edgecolor='black',lw=0.8)
        ax.errorbar(xp, ates, yerr=[1.96*s for s in ses],
                    fmt='none',color='black',capsize=4,lw=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.capitalize() for c in cats_order])
    ax.set_title(f"Treatment: {wt} (discount>{'5' if wt=='W5' else '10'}%)")
    ax.set_ylabel("ATE (log-units)")
    ax.legend(title="Learner"); ax.grid(True,alpha=0.3,axis='y')
    ax.axhline(0,color='black',lw=0.8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step04_grouped_dml_v2.png",dpi=150,bbox_inches='tight')
plt.close()
print("\nSaved: grouped_dml_v2_noprice.csv + figure")
