"""Step 8: Seven-bin causal forest (D1-D7 ordinal treatment)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.stats as st

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step08_sevenbin"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
BIN_LABELS= ["D1\n(0-5%)","D2\n(5-10%)","D3\n(10-20%)","D4\n(20-30%)",
              "D5\n(30-40%)","D6\n(40-50%)","D7\n(≥50%)"]
BIN_MAP   = {"D1":1,"D2":2,"D3":3,"D4":4,"D5":5,"D6":6,"D7":7}

def load_and_build(cat, n_rg=1, max_rows=4000):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","discount_depth","depth_bin",
            "unit_price","week_of_year","brand_code_uc"]
    avail = pf_.schema_arrow.names
    cols = [c for c in want if c in avail]
    df = pd.concat([pf_.read_row_group(i,columns=cols).to_pandas()
                    for i in range(min(n_rg,pf_.num_row_groups))],ignore_index=True)
    df = df.dropna(subset=["log_units","depth_bin"]).head(max_rows).copy()
    df["depth_num"] = df["depth_bin"].map(BIN_MAP).fillna(1).astype(int)
    df["log_price"] = np.log(df["unit_price"].clip(0.01))
    df["sin_wk"]   = np.sin(2*np.pi*df["week_of_year"].fillna(26)/52)
    df["cos_wk"]   = np.cos(2*np.pi*df["week_of_year"].fillna(26)/52)
    df["brand_h"]  = df["brand_code_uc"].astype(str).apply(lambda x: hash(x)%50/50)
    df["upc_h"]    = df["upc"].astype(str).apply(lambda x: hash(x)%50/50)
    return df

def bin_dml_ate(df, bin_num, ref_bin=1, max_per=500, seed=42):
    """
    Estimate ATE of being in depth bin `bin_num` vs ref_bin=1 (D1=not promoted).
    DML: Y - Y_hat ~ (W - W_hat) where W = 1{depth_bin==bin_num}.
    """
    # Keep only obs in bin_num or ref_bin
    sub = df[df["depth_num"].isin([ref_bin, bin_num])].copy()
    if len(sub) < 30: return np.nan, np.nan, np.nan, 0
    
    W = (sub["depth_num"]==bin_num).astype(float).values
    Y = sub["log_units"].values
    X = sub[["sin_wk","cos_wk","log_price","brand_h","upc_h"]].fillna(0).values
    
    # 5-fold cross-fit
    n = len(sub)
    rng = np.random.default_rng(seed+bin_num)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    
    Y_res = np.zeros(n); W_res = np.zeros(n)
    sc = StandardScaler()
    for fi,test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(5) if j!=fi])
        Xtr = sc.fit_transform(X[train_idx])
        Xte = sc.transform(X[test_idx])
        my = Ridge(1.0).fit(Xtr,Y[train_idx])
        mw = Ridge(1.0).fit(Xtr,W[train_idx])
        Y_res[test_idx] = Y[test_idx]-my.predict(Xte)
        W_res[test_idx] = W[test_idx]-mw.predict(Xte)
    
    # Wald
    denom = np.sum(W_res**2)
    if denom < 1e-10: return np.nan, np.nan, np.nan, n
    theta = np.sum(W_res*Y_res)/denom
    
    # SE: Neyman-orthogonal
    eps = Y_res*W_res - theta*W_res**2
    se = np.sqrt(np.var(eps,ddof=1)*n)/denom
    z = theta/se if se>0 else np.nan
    p = float(2*(1-st.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
    return theta, se, p, n

all_rows = []

fig, axes = plt.subplots(1,3,figsize=(17,5))
fig.suptitle("Step 8 — Seven-Bin DML Dose-Response (D1–D7 vs. reference D1)\n"
             "(Cross-fit DML per bin; Neyman-orthogonal SE; 95% CI; N_RG=1)",
             fontsize=11,fontweight='bold')
cc = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}

for ax,cat in zip(axes,CATS):
    print(f"\n{'='*45}\n  {cat.upper()}")
    df = load_and_build(cat,1,4000)
    
    dist = df["depth_num"].value_counts().sort_index()
    print(f"  n={len(df):,}  Bin distribution:")
    for d,c in dist.items(): print(f"    D{d}: {c:,} ({100*c/len(df):.1f}%)")
    
    thetas,ses,ps,ns=[],[],[],[]
    for b in range(1,8):
        th,se,p,n = bin_dml_ate(df,b,ref_bin=1)
        thetas.append(th); ses.append(se); ps.append(p); ns.append(n)
        sig="***" if (p is not None and not np.isnan(p) and p<0.001) else \
            ("**" if (p is not None and not np.isnan(p) and p<0.01) else \
             ("*" if (p is not None and not np.isnan(p) and p<0.05) else ""))
        print(f"    D{b}: θ={th:.4f}  SE={se:.4f}  p={p:.4f}  n={n}  {sig}" if not np.isnan(th) else f"    D{b}: insufficient obs")
        all_rows.append({"category":cat,"bin":f"D{b}","bin_num":b,
                         "n_obs":n,"theta":round(th,5) if not np.isnan(th) else np.nan,
                         "se":round(se,5) if not np.isnan(se) else np.nan,
                         "ci_lo":round(th-1.96*se,5) if not np.isnan(th) else np.nan,
                         "ci_hi":round(th+1.96*se,5) if not np.isnan(th) else np.nan,
                         "p":round(p,5) if (p is not None and not np.isnan(p)) else np.nan,
                         "sig":sig})
    
    # Plot dose-response
    bins = [f"D{b}" for b in range(1,8)]
    col = cc[cat]
    valid = [(i,th,se) for i,(th,se) in enumerate(zip(thetas,ses)) if not np.isnan(th)]
    
    if valid:
        xi=[v[0] for v in valid]; tv=[v[1] for v in valid]; sv=[v[2] for v in valid]
        # Reference D1 = 0 by construction
        ax.axhline(0,color='gray',lw=1.5,ls='--',label="D1 reference",zorder=1)
        ax.fill_between(xi,[t-1.96*s for t,s in zip(tv,sv)],
                        [t+1.96*s for t,s in zip(tv,sv)],
                        color=col,alpha=0.2,zorder=2)
        ax.plot(xi,tv,color=col,lw=2.5,marker='o',ms=7,zorder=3,label=cat.capitalize())
        ax.scatter([v[0] for v in valid if abs(ps[v[0]] or 1)<0.05],
                   [v[1] for v in valid if abs(ps[v[0]] or 1)<0.05],
                   color='red',s=60,zorder=4,label="p<0.05")
    
    ax.set_xticks(range(7))
    ax.set_xticklabels(BIN_LABELS,fontsize=8)
    ax.set_title(f"{cat.capitalize()}",fontsize=12,fontweight='bold')
    ax.set_xlabel("Promotion Depth Bin")
    ax.set_ylabel("ATE vs. D1 (log-units)")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step08_sevenbin_doseresponse.png",dpi=150,bbox_inches='tight')
plt.close()

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/sevenbin_dml_results.csv",index=False)
print("\n=== SEVEN-BIN SUMMARY ===")
print(out[["category","bin","theta","se","p","sig"]].to_string(index=False))
print("\nFigure saved.")
