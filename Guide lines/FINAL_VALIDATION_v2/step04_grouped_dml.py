"""
Step 4: Full GroupKFold DML with Lasso, ElasticNet, Random Forest, GBM.
GroupKFold by store — no store leakage.
Store-clustered Liang-Zeger SEs.
W5 = 1{depth>0.05}, W10 = 1{depth>=0.10}
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as st, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step04_grouped_dml"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS = ["cereal","snacks","coffee"]; N_RG = 3

LEARNER_SPECS = [
    ("Lasso",        Pipeline([("sc", StandardScaler()),
                               ("m",  Lasso(alpha=0.001, max_iter=5000))])),
    ("ElasticNet",   Pipeline([("sc", StandardScaler()),
                               ("m",  ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000))])),
    ("RandomForest", RandomForestRegressor(n_estimators=50, max_depth=5,
                                           min_samples_leaf=20, n_jobs=-1, random_state=42)),
    ("GBM",          GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                               min_samples_leaf=20, random_state=42)),
]

def lz_clustered_se(Y_res, W_res, groups):
    """Liang-Zeger store-clustered SE."""
    G = len(np.unique(groups)); n = len(Y_res)
    theta = float(np.sum(W_res*Y_res) / np.sum(W_res**2))
    eps   = Y_res - theta * W_res
    df_   = pd.DataFrame({"s": groups, "Wr": W_res, "ep": eps})
    agg   = df_.groupby("s").apply(lambda x: float(np.sum(x["Wr"]*x["ep"]))).values
    meat  = np.sum(agg**2) * (G/(G-1)) * ((n-1)/(n-3))
    bread = np.sum(W_res**2)
    se_c  = float(np.sqrt(max(0, meat) / bread**2))
    t_    = theta / se_c
    p_    = float(2*(1-st.norm.cdf(abs(t_))))
    return theta, se_c, t_, p_

def run_dml_cat(cat):
    print(f"\n  {cat.upper()}")
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    cols = ["store_code_uc","upc","log_units","discount_depth",
            "W5","W10","week_of_year","brand_code_uc","unit_price"]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas()
              for i in range(min(N_RG, pf_.num_row_groups))]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["log_units"])
    
    # Feature matrix X (store/UPC/time controls — no numeric retailer)
    # Use available columns only
    df["upc_hash"]   = pd.factorize(df["upc"])[0]
    df["store_hash"] = pd.factorize(df["store_code_uc"])[0]
    df["brand_hash"] = pd.factorize(df["brand_code_uc"])[0]
    
    feat_cols = ["upc_hash","brand_hash","week_of_year","unit_price"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    
    X = df[feat_cols].fillna(0).values
    Y = df["log_units"].values
    groups = df["store_code_uc"].values
    
    n_splits = min(5, len(np.unique(groups)))
    gkf  = GroupKFold(n_splits=n_splits)
    cv_splits = list(gkf.split(X, Y, groups))
    
    print(f"  n={len(df):,} | n_stores={len(np.unique(groups))} | splits={n_splits}")
    
    results = []
    
    for W_name in ["W5","W10"]:
        if W_name not in df.columns:
            # Reconstruct
            df["W5"]  = (df["discount_depth"] > 0.05).astype(int)
            df["W10"] = (df["discount_depth"] >= 0.10).astype(int)
        
        W = df[W_name].values.astype(float)
        
        print(f"  Treatment: {W_name} (mean={W.mean():.3f})")
        
        for lr_name, lr in LEARNER_SPECS:
            try:
                # Cross-fitted predictions using IDENTICAL folds
                Y_hat = cross_val_predict(lr, X, Y, cv=cv_splits)
                W_hat = cross_val_predict(lr, X, W, cv=cv_splits)
                Y_res = Y - Y_hat
                W_res = W - W_hat
                
                theta, se, t_stat, p_val = lz_clustered_se(Y_res, W_res, groups)
                
                sig = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else ""))
                print(f"    {lr_name:12s} | ATE={theta:.4f} se={se:.5f} t={t_stat:.2f} {sig}")
                
                results.append({
                    "category": cat, "treatment": W_name, "learner": lr_name,
                    "ate": round(theta,5), "se_clustered": round(se,5),
                    "t_stat": round(t_stat,2), "p_val": round(p_val,5),
                    "sig": sig, "n_obs": len(df), "n_stores": len(np.unique(groups)),
                    "n_splits": n_splits
                })
            except Exception as e:
                print(f"    {lr_name}: ERROR {e}")
    
    return results

all_results = []
for cat in CATS:
    all_results.extend(run_dml_cat(cat))

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUT_DIR}/grouped_dml_production.csv", index=False)

print(f"\n{'='*65}")
print("FULL GROUPED DML RESULTS:")
print(res_df[["category","treatment","learner","ate","se_clustered","t_stat","sig"]].to_string(index=False))

# ── Summary table: mean ATE across learners ──────────────────────────────
avg = (res_df.groupby(["category","treatment"])
       .agg(ate_mean=("ate","mean"), ate_min=("ate","min"),
            ate_max=("ate","max"), n_learners=("learner","count"))
       .reset_index())
avg.to_csv(f"{OUT_DIR}/grouped_dml_summary.csv", index=False)
print(f"\nAVERAGE ATE BY CATEGORY + TREATMENT:")
print(avg.to_string(index=False))

# Check ordering: Cereal > Snacks > Coffee for W5
order_check = avg[avg["treatment"]=="W5"].sort_values("ate_mean",ascending=False)
print(f"\nOrdering (W5): {' > '.join(order_check['category'].str.capitalize().tolist())}")

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Step 4 — Grouped DML (GroupKFold by Store)\nATE by Learner, Treatment, and Category",
             fontsize=12, fontweight='bold')

colors = {"Lasso":"steelblue","ElasticNet":"darkorange",
          "RandomForest":"forestgreen","GBM":"crimson"}
markers = {"cereal":"o","snacks":"s","coffee":"^"}

for ax_i, W_name in enumerate(["W5","W10"]):
    ax = axes[ax_i]
    sub = res_df[res_df["treatment"]==W_name]
    
    x = np.arange(len(CATS))
    width = 0.18
    learners = [s[0] for s in LEARNER_SPECS]
    
    for j, lr in enumerate(learners):
        sub_lr = sub[sub["learner"]==lr]
        ates = [sub_lr[sub_lr["category"]==cat]["ate"].values[0] 
                if len(sub_lr[sub_lr["category"]==cat])>0 else np.nan
                for cat in CATS]
        ses  = [sub_lr[sub_lr["category"]==cat]["se_clustered"].values[0]
                if len(sub_lr[sub_lr["category"]==cat])>0 else 0.0
                for cat in CATS]
        
        offset = (j - 1.5) * width
        ax.bar(x + offset, ates, width*0.85, yerr=ses,
               label=lr, color=colors[lr], alpha=0.8,
               capsize=3, error_kw=dict(linewidth=1.2))
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CATS], fontsize=11)
    ax.set_ylabel("ATE (log units per unit increase in treatment)")
    ax.set_title(f"Treatment: {W_name} (=1{{depth{'>' if W_name=='W5' else '>='}"
                 f"{'5' if W_name=='W5' else '10'}%}})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, sub["ate"].max()*1.3 if len(sub)>0 else 1.0)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step04_grouped_dml.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved.")
