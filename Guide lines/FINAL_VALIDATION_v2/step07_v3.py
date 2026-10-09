"""Step 7 v3: Honest GATES — clean quartile assignment."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]

def load_and_build(cat, n_rg=1, max_rows=3000):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","W5","unit_price","week_of_year","brand_code_uc","depth_bin"]
    avail = pf_.schema_arrow.names
    cols = [c for c in want if c in avail]
    df = pd.concat([pf_.read_row_group(i,columns=cols).to_pandas()
                    for i in range(min(n_rg,pf_.num_row_groups))],ignore_index=True)
    df = df.dropna(subset=["log_units","W5"]).head(max_rows).copy()
    db_map = {"D1":1,"D2":2,"D3":3,"D4":4,"D5":5,"D6":6,"D7":7}
    df["depth_num"] = df["depth_bin"].map(db_map).fillna(1).astype(int) if "depth_bin" in df.columns else 1
    df["log_price"] = np.log(df["unit_price"].clip(0.01))
    df["sin_wk"]   = np.sin(2*np.pi*df["week_of_year"].fillna(26)/52)
    df["brand_h"]  = df["brand_code_uc"].astype(str).apply(lambda x: hash(x)%50/50)
    df["upc_h"]    = df["upc"].astype(str).apply(lambda x: hash(x)%50/50)
    X = df[["sin_wk","log_price","brand_h","upc_h","depth_num"]].fillna(0).values
    Y = df["log_units"].values
    W = df["W5"].astype(float).values
    return X, Y, W

def rank_quartile(arr, rng):
    """Assign 0-3 groups by rank, breaking ties randomly."""
    n = len(arr)
    noise = rng.uniform(0,1e-8,n)
    ranks = np.argsort(np.argsort(arr + noise))  # stable sort ranking
    cutoffs = np.array([n*0.25, n*0.50, n*0.75]).astype(int)
    groups = np.zeros(n, dtype=int)
    groups[ranks >= cutoffs[0]] = 1
    groups[ranks >= cutoffs[1]] = 2
    groups[ranks >= cutoffs[2]] = 3
    return groups

def wald_gate(Wr, Yr, mask):
    wr = Wr[mask]; yr = Yr[mask]
    d = np.sum(wr**2)
    if d < 1e-10 or mask.sum() < 5: return np.nan, np.nan
    th = np.sum(wr*yr)/d
    eps = yr*wr - th*wr**2
    se = np.sqrt(np.var(eps,ddof=1)*mask.sum())/d
    return th, se

def honest_gates(X,Y,W,n_perm=200,seed=42):
    rng = np.random.default_rng(seed)
    n = len(Y); half = n//2
    idx = rng.permutation(n)
    A,B = idx[:half],idx[half:]
    sc = StandardScaler()
    Xa = sc.fit_transform(X[A]); Xb = sc.transform(X[B])
    my = Ridge(1.0).fit(Xa,Y[A]); mw = Ridge(1.0).fit(Xa,W[A])
    Yr = Y[B]-my.predict(Xb)
    Wr = W[B]-mw.predict(Xb)
    
    # CATE proxy: depth_num + noise to break ties
    depth_B = X[B,-1].copy()  # depth_num (last feature)
    q_groups = rank_quartile(depth_B, rng)
    
    gv,gs = {},{}
    for g in range(4):
        m=(q_groups==g)
        th,se = wald_gate(Wr,Yr,m)
        gv[g]=th; gs[g]=se
    
    obs_sp = gv[3]-gv[0] if (not np.isnan(gv[3]) and not np.isnan(gv[0])) else np.nan
    
    perm_sp=[]
    for _ in range(n_perm):
        Wp=W.copy(); rng.shuffle(Wp)
        mwp=Ridge(1.0).fit(Xa,Wp[A])
        Wrp=Wp[B]-mwp.predict(Xb)
        pvg={}
        for g in range(4):
            m=(q_groups==g); th,_ = wald_gate(Wrp,Yr,m); pvg[g]=th
        if not (np.isnan(pvg[3]) or np.isnan(pvg[0])):
            perm_sp.append(pvg[3]-pvg[0])
    
    perm_arr=np.array(perm_sp)
    pval = float(np.mean(np.abs(perm_arr)>=abs(obs_sp))) if len(perm_arr)>0 and not np.isnan(obs_sp) else np.nan
    return gv,gs,obs_sp,perm_arr,pval

all_res=[]
fig,axes=plt.subplots(1,3,figsize=(17,5))
fig.suptitle("Step 7 — Honest GATES + 200-Permutation Test\n"
             "(50/50 split; rank-based quartile proxy=depth_bin; Neyman orthog.; W=W5)",
             fontsize=11,fontweight='bold')
cc={"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}

for ax,cat in zip(axes,CATS):
    print(f"\n{'='*45}\n  {cat.upper()}")
    X,Y,W = load_and_build(cat,1,3000)
    print(f"  n={len(Y)}, W=1 rate={W.mean():.3f}")
    
    gv,gs,obs_sp,perm_arr,pval = honest_gates(X,Y,W)
    
    for g in range(4):
        print(f"    Q{g+1}: GATE={gv[g]:.4f} SE={gs[g]:.4f}")
    print(f"  Spread Q4-Q1={obs_sp:.4f}  perm_p={pval:.4f}  n_perm={len(perm_arr)}")
    
    sig="***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else "n.s."))
    all_res.append({"category":cat,"n_obs":len(Y),
                    "gate_q1":round(gv[0],4) if not np.isnan(gv[0]) else np.nan,
                    "se_q1":round(gs[0],4) if not np.isnan(gs[0]) else np.nan,
                    "gate_q2":round(gv[1],4) if not np.isnan(gv[1]) else np.nan,
                    "gate_q3":round(gv[2],4) if not np.isnan(gv[2]) else np.nan,
                    "gate_q4":round(gv[3],4) if not np.isnan(gv[3]) else np.nan,
                    "se_q4":round(gs[3],4) if not np.isnan(gs[3]) else np.nan,
                    "obs_spread":round(obs_sp,4) if not np.isnan(obs_sp) else np.nan,
                    "n_perm":len(perm_arr),"perm_pval":round(pval,4) if not np.isnan(pval) else np.nan,
                    "sig":sig})
    
    # Figure
    gvals=[gv.get(g,np.nan) for g in range(4)]
    gserr=[gs.get(g,0) for g in range(4)]
    col=cc[cat]; alph=[0.35,0.55,0.75,1.0]
    for i,(qn,gval,gse,a) in enumerate(zip(["Q1","Q2","Q3","Q4"],gvals,gserr,alph)):
        if np.isnan(gval): continue
        ax.bar(qn,gval,color=col,alpha=a,edgecolor='black',lw=1.2,zorder=3)
        ax.errorbar(qn,gval,yerr=1.96*gse,fmt='none',color='black',capsize=5,lw=2,zorder=4)
    ax.axhline(0,color='gray',lw=1,ls='--')
    ax.set_title(f"{cat.capitalize()}\nQ4−Q1={obs_sp:.3f}  p={pval:.3f} ({sig})",fontsize=11)
    ax.set_xlabel("Depth Quartile Group (Q1=shallow, Q4=deep)"); ax.set_ylabel("GATE (log-units)")
    ax.grid(True,alpha=0.3,axis='y')
    
    axi=ax.inset_axes([0.60,0.52,0.38,0.42])
    axi.hist(perm_arr,bins=30,color='gray',alpha=0.7,density=True,zorder=2)
    axi.axvline(obs_sp,color='red',lw=2)
    axi.set_title("Perm. null",fontsize=7); axi.tick_params(labelsize=6)
    axi.text(0.05,0.85,f"p={pval:.3f}",transform=axi.transAxes,fontsize=6)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step07_honest_gates.png",dpi=150,bbox_inches='tight')
plt.close()

out=pd.DataFrame(all_res)
out.to_csv(f"{OUT_DIR}/honest_gates_permtest.csv",index=False)
print("\n=== GATES SUMMARY ===")
print(out[["category","gate_q1","gate_q4","obs_spread","perm_pval","sig"]].to_string(index=False))
print("\nFigure saved.")
