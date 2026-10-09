"""Step 8 v4: Per-bin honest forests with IJ-based ATE SE (not SD/sqrt(n)).
Fix: use inf_.stderr_mean from econml InferenceResults (IJ variance for ATE).
Fix: use a common D1 pool across all Dk comparisons (shared target population).
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from econml.dml import CausalForestDML

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step08_causal_forest_dose_response"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
N_D1      = 6000   # common D1 pool size (fixed across all Dk)
N_DK      = 6000   # Dk pool per bin

def load_rms(cat):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["log_units","depth_bin","week_of_year","regular_price_q75","brand_code_uc"]
    have = [c for c in want if c in pf.schema_arrow.names]
    frames = [pf.read_row_group(i, columns=have).to_pandas()
              for i in range(min(1, pf.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(int)
    return df

all_rows = []
for cat in CATS:
    print(f"\n{'='*50}\n  {cat.upper()}")
    df = load_rms(cat).dropna(subset=["log_units","depth_num"])
    
    # Common D1 pool (SAME across all Dk comparisons)
    d1_all = df[df["depth_num"]==1].copy()
    n_d1   = min(N_D1, len(d1_all))
    d1_pool= d1_all.sample(n_d1, random_state=7)
    print(f"  Common D1 pool: {n_d1}")

    for k in range(2, 8):
        dk_all = df[df["depth_num"]==k].copy()
        if len(dk_all) < 30:
            print(f"  D{k}: too sparse ({len(dk_all)}) — skip"); continue
        n_dk = min(N_DK, len(dk_all))
        dk_pool = dk_all.sample(n_dk, random_state=k*100+7, replace=len(dk_all)<n_dk)
        
        sub = pd.concat([d1_pool, dk_pool], ignore_index=True)
        sub = sub.sample(frac=1, random_state=42).reset_index(drop=True)
        
        Y = sub["log_units"].values
        W = (sub["depth_num"]==k).astype(float).values
        wk= sub["week_of_year"].fillna(26).values.astype(float)
        X_cols = [np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52)]
        if "regular_price_q75" in sub.columns:
            X_cols.append(sub["regular_price_q75"].fillna(sub["regular_price_q75"].median()).values)
        if "brand_code_uc" in sub.columns:
            X_cols.append(pd.Categorical(sub["brand_code_uc"]).codes.astype(float))
        X = np.column_stack(X_cols)
        
        rng = np.random.default_rng(42+k)
        n = len(Y); idx = np.arange(n); rng.shuffle(idx)
        ia, ib = idx[:n//2], idx[n//2:]
        
        try:
            cf = CausalForestDML(model_y=Ridge(alpha=1.0), model_t=Ridge(alpha=1.0),
                                 n_estimators=100, min_samples_leaf=5,
                                 random_state=42, honest=True, inference=True)
            cf.fit(Y[ia], W[ia], X=X[ia])
            inf_ = cf.effect_inference(X[ib])
            # IJ-based ATE and SE (NOT SD(CATE)/sqrt(n))
            ate = float(inf_.mean_point)
            se  = float(inf_.stderr_mean)   # IJ variance for mean
            z   = ate/se if se>0 else np.nan
            sig = "***" if abs(z)>3.29 else ("**" if abs(z)>2.58 else ("*" if abs(z)>1.96 else "n.s."))
            print(f"  D1vsD{k}: ate={ate:.4f} se_IJ={se:.4f} z={z:.2f}{sig}  n1={n_d1} nk={n_dk}")
        except Exception as e:
            # Fallback: try mean_point via alternate call
            try:
                inf2_ = cf.effect_inference(X[ib])
                ate = float(np.mean(inf2_.point_estimate))
                # SE from individual pointwise IJ SEs (unbiased for uncorrelated obs)
                se  = float(np.sqrt(np.mean(inf2_.stderr**2)) / np.sqrt(len(inf2_.stderr)))
                z   = ate/se if se>0 else np.nan
                sig = "***" if abs(z)>3.29 else ("**" if abs(z)>2.58 else ("*" if abs(z)>1.96 else "n.s."))
                print(f"  D1vsD{k}: ate={ate:.4f} se_IJ_fb={se:.4f} z={z:.2f}{sig}  [fallback]")
            except Exception as e2:
                ate=se=z=np.nan; sig="err"
                print(f"  D1vsD{k}: ERROR {e} | {e2}")

        all_rows.append({"category":cat,"bin":k,"n_d1":n_d1,"n_dk":n_dk,
                         "ate":round(ate,4) if not np.isnan(ate) else np.nan,
                         "se_IJ":round(se,4) if not np.isnan(se) else np.nan,
                         "z":round(z,3) if not np.isnan(z) else np.nan,"sig":sig,
                         "inference":"IJ-forest"})

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/dose_response_v4_IJ.csv", index=False)

# Figure
fig, axes = plt.subplots(1,3,figsize=(16,5),sharey=False)
CC = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
LABELS={2:"D2\n5-10%",3:"D3\n10-20%",4:"D4\n20-30%",5:"D5\n30-40%",6:"D6\n40-50%",7:"D7\n≥50%"}
for ax,cat in zip(axes,CATS):
    r=out[out["category"]==cat].dropna(subset=["ate"])
    xs=r["bin"].values; ys=r["ate"].values; ses=r["se_IJ"].values; sigs=r["sig"].values
    ax.bar(xs,ys,color=CC[cat],alpha=0.65,width=0.55,zorder=3)
    ax.errorbar(xs,ys,yerr=1.96*ses,fmt='none',color='k',capsize=5,lw=1.5,zorder=5)
    for xi,yi,se_,sig_ in zip(xs,ys,ses,sigs):
        ax.text(xi,yi+(0.03 if yi>=0 else -0.06),sig_,ha='center',fontsize=8,fontweight='bold')
    ax.axhline(0,color='k',lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([LABELS.get(int(x),"?") for x in xs],fontsize=7.5)
    ax.set_title(f"{cat.capitalize()}\nD1 vs Dk (IJ-based SE; common D1 pool)",fontsize=10,fontweight='bold')
    ax.set_xlabel("Depth bin vs D1"); ax.set_ylabel("ATE (Δ log units)"); ax.grid(True,alpha=0.3)
plt.suptitle("Step 8 v4 — Per-Bin Forests: D1 vs Dk (IJ-based ATE SE; common D1 pool)\n"
             "SE = IJ variance for mean, NOT SD(CATE_i)/sqrt(n)",
             fontsize=11,fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step08_perbin_v4_IJ.png",dpi=150,bbox_inches='tight')
plt.close()
print("\n✓ Step 8 v4 done")
print(out[["category","bin","n_d1","n_dk","ate","se_IJ","z","sig"]].to_string(index=False))
