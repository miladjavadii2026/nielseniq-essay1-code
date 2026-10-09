"""Step 4 fast: Lasso + ElasticNet + Ridge only (RF/GBM need workstation)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as st, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step04_grouped_dml"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS = ["cereal","snacks","coffee"]; N_RG = 2

LEARNERS = [
    ("Ridge",      Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=1.0))])),
    ("Lasso",      Pipeline([("sc",StandardScaler()),("m",Lasso(alpha=0.001,max_iter=3000))])),
    ("ElasticNet", Pipeline([("sc",StandardScaler()),("m",ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=3000))])),
]

def lz_se(Y_res, W_res, groups):
    G = len(np.unique(groups)); n = len(Y_res)
    theta = float(np.sum(W_res*Y_res) / np.sum(W_res**2))
    eps = Y_res - theta * W_res
    df_ = pd.DataFrame({"s":groups,"Wr":W_res,"ep":eps})
    agg = df_.groupby("s").apply(lambda x: float(np.sum(x["Wr"]*x["ep"]))).values
    meat = np.sum(agg**2)*(G/(G-1))*((n-1)/(n-3))
    se = float(np.sqrt(max(0,meat)/np.sum(W_res**2)**2))
    t_ = theta/se; p_ = float(2*(1-st.norm.cdf(abs(t_))))
    return theta,se,t_,p_

all_results = []

for cat in CATS:
    print(f"\n{'='*50}\n  {cat.upper()}")
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    cols = ["store_code_uc","upc","log_units","discount_depth",
            "week_of_year","brand_code_uc","unit_price"]
    df = pd.concat([pf_.read_row_group(i,columns=cols).to_pandas()
                    for i in range(min(N_RG,pf_.num_row_groups))],ignore_index=True)
    df = df.dropna(subset=["log_units"])
    df["W5"]  = (df["discount_depth"]>0.05).astype(float)
    df["W10"] = (df["discount_depth"]>=0.10).astype(float)
    df["upc_h"]   = pd.factorize(df["upc"])[0].astype(float)
    df["brand_h"] = pd.factorize(df["brand_code_uc"])[0].astype(float)
    df["unit_price"] = df["unit_price"].fillna(0)
    
    X = df[["upc_h","brand_h","week_of_year","unit_price"]].fillna(0).values
    Y = df["log_units"].values
    groups = df["store_code_uc"].values
    n_sp = min(5, len(np.unique(groups)))
    cv = list(GroupKFold(n_splits=n_sp).split(X, Y, groups))
    print(f"  n={len(df):,} stores={len(np.unique(groups))} splits={n_sp}")
    
    for W_name in ["W5","W10"]:
        W = df[W_name].values
        for lr_name, lr in LEARNERS:
            Y_hat = cross_val_predict(lr, X, Y, cv=cv)
            W_hat = cross_val_predict(lr, X, W, cv=cv)
            theta,se,t_,p_ = lz_se(Y-Y_hat, W-W_hat, groups)
            sig = "***" if p_<0.001 else ("**" if p_<0.01 else ("*" if p_<0.05 else ""))
            print(f"    {W_name} {lr_name:10s}: ATE={theta:.4f} se={se:.5f} t={t_:.2f} {sig}")
            all_results.append({
                "category":cat,"treatment":W_name,"learner":lr_name,
                "ate":round(theta,5),"se_clustered":round(se,5),
                "t_stat":round(t_,2),"p_val":round(p_,5),"sig":sig,
                "n_obs":len(df),"n_stores":len(np.unique(groups))
            })

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUT_DIR}/grouped_dml_production.csv", index=False)

# Ordering check for W5
w5 = res_df[res_df["treatment"]=="W5"].groupby("category")["ate"].mean().reset_index()
w5 = w5.sort_values("ate",ascending=False)
print(f"\nW5 ATE ordering: {' > '.join(w5['category'].str.capitalize())}")
note = f"Correct ordering: {' > '.join(w5['category'].str.capitalize())} (matches data)"

# Summary
avg = (res_df.groupby(["category","treatment"])
       .agg(ate_mean=("ate","mean"),ate_std=("ate","std")).reset_index())
avg.to_csv(f"{OUT_DIR}/grouped_dml_summary.csv", index=False)
print("\nSUMMARY:")
print(avg.to_string(index=False))

# Figure
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Step 4 — Grouped DML (GroupKFold by Store)\nRidge, Lasso, ElasticNet Learners",
             fontsize=12, fontweight='bold')
cols_lr = {"Ridge":"steelblue","Lasso":"darkorange","ElasticNet":"forestgreen"}

for ax_i, W_name in enumerate(["W5","W10"]):
    ax = axes[ax_i]
    sub = res_df[res_df["treatment"]==W_name]
    x = np.arange(3); width = 0.22
    for j,(lr_name,_) in enumerate(LEARNERS):
        s = sub[sub["learner"]==lr_name]
        ates = [s[s["category"]==c]["ate"].values[0] if len(s[s["category"]==c])>0 else np.nan
                for c in CATS]
        ses  = [s[s["category"]==c]["se_clustered"].values[0] if len(s[s["category"]==c])>0 else 0
                for c in CATS]
        ax.bar(x+(j-1)*width, ates, width*0.85, yerr=ses,
               label=lr_name, color=cols_lr[lr_name], alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CATS], fontsize=11)
    ax.set_ylabel("ATE (log units)")
    W_def = "depth>5%" if W_name=="W5" else "depth≥10%"
    ax.set_title(f"{W_name}: 1{{{W_def}}}")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis='y')
    ax.set_ylim(0, sub["ate"].max()*1.4+0.01)
    # Note correct ordering
    order_str = " > ".join(w5["category"].str.capitalize())
    ax.text(0.05, 0.95, f"Ordering: {order_str}", transform=ax.transAxes,
            fontsize=8, va='top', style='italic')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step04_grouped_dml.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")
