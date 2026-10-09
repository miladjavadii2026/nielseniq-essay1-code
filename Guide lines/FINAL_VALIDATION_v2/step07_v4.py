"""Step 7 v4: Honest GATES using CausalForestDML CATE predictions.
Fixes:
- CATE proxy from held-out CausalForestDML predictions (pre-treatment X only, no depth)
- p-value formula: (1 + count) / (1 + n_perm) — Phipson & Smyth 2010 correction
- 200 permutations with corrected formula
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]

def load_and_build(cat, n_rg=1, max_rows=2000):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","W5","week_of_year","brand_code_uc","depth_bin"]
    avail = pf_.schema_arrow.names
    cols  = [c for c in want if c in avail]
    df = pd.concat([pf_.read_row_group(i,columns=cols).to_pandas()
                    for i in range(min(n_rg,pf_.num_row_groups))],ignore_index=True)
    df = df.dropna(subset=["log_units","W5"]).head(max_rows).copy()
    # Pre-treatment features ONLY — no unit_price, no depth_bin
    df["brand_h"] = df["brand_code_uc"].astype(str).apply(lambda x: hash(x)%200/200)
    df["upc_h"]   = df["upc"].astype(str).apply(lambda x: hash(x)%500/500)
    df["sin_wk"]  = np.sin(2*np.pi*df["week_of_year"].fillna(26)/52)
    df["cos_wk"]  = np.cos(2*np.pi*df["week_of_year"].fillna(26)/52)
    X = df[["brand_h","upc_h","sin_wk","cos_wk"]].fillna(0).values
    Y = df["log_units"].values
    W = df["W5"].astype(float).values
    return X, Y, W

def rank_q4(arr, rng):
    """Rank-based balanced quartile assignment (breaks ties via noise)."""
    noise = rng.uniform(0, 1e-8, len(arr))
    ranks = np.argsort(np.argsort(arr + noise))
    n = len(arr)
    g = np.zeros(n, dtype=int)
    g[ranks >= n*0.25] = 1
    g[ranks >= n*0.50] = 2
    g[ranks >= n*0.75] = 3
    return g

def wald_gate(Wr, Yr, mask):
    wr=Wr[mask]; yr=Yr[mask]
    d=np.sum(wr**2)
    if d<1e-10 or mask.sum()<5: return np.nan,np.nan
    th=np.sum(wr*yr)/d
    eps=yr*wr-th*wr**2
    se=np.sqrt(np.var(eps,ddof=1)*mask.sum())/d
    return th,se

def honest_gates_cfml(X, Y, W, n_perm=200, seed=42):
    """
    1. 50/50 honest split
    2. Fit CausalForestDML on A (pre-treatment X, no depth)
    3. Predict CATE on B → form quartile groups
    4. Estimate GATEs via Neyman-orthogonal scores in B
    5. Permutation: (1 + count) / (1 + n_perm) [Phipson & Smyth 2010]
    """
    rng = np.random.default_rng(seed)
    n = len(Y); half = n//2
    idx = rng.permutation(n)
    A, B = idx[:half], idx[half:]
    
    Xa,Ya,Wa = X[A],Y[A],W[A]
    Xb,Yb,Wb = X[B],Y[B],W[B]
    
    # --- Step 1: CausalForestDML on A ---
    cf = CausalForestDML(
        model_y=Ridge(alpha=1.0), model_t=Ridge(alpha=1.0),
        n_estimators=50, min_samples_leaf=5,
        random_state=seed, honest=True,
        inference=False  # speed
    )
    cf.fit(Ya, Wa, X=Xa)
    
    # --- Step 2: Predict CATEs on B (using pre-treatment X only) ---
    cate_B = cf.effect(Xb).flatten()  # shape (n_B,)
    
    # --- Step 3: Form quartile groups from predicted CATE ---
    q_groups = rank_q4(cate_B, rng)
    
    # --- Step 4: Neyman-orthogonal Wald GATEs on B ---
    # Nuisance from A for use in B
    sc = StandardScaler()
    Xa_sc = sc.fit_transform(Xa); Xb_sc = sc.transform(Xb)
    my = Ridge(1.0).fit(Xa_sc, Ya); mw = Ridge(1.0).fit(Xa_sc, Wa)
    Yr_B = Yb - my.predict(Xb_sc)
    Wr_B = Wb - mw.predict(Xb_sc)
    
    gv,gs = {},{}
    for g in range(4):
        m=(q_groups==g); th,se=wald_gate(Wr_B,Yr_B,m); gv[g]=th; gs[g]=se
    
    obs_sp = (gv[3]-gv[0]) if (not np.isnan(gv[3]) and not np.isnan(gv[0])) else np.nan
    
    # --- Step 5: Permutation test ---
    perm_sp = []
    for _ in range(n_perm):
        Wp = W.copy(); rng.shuffle(Wp)
        # Refit nuisance W on A with shuffled W, recompute Wr_B
        mwp = Ridge(1.0).fit(Xa_sc, Wp[A])
        Wr_Bp = Wp[B] - mwp.predict(Xb_sc)
        pvg = {}
        for g in range(4):
            m=(q_groups==g); th,_=wald_gate(Wr_Bp,Yr_B,m); pvg[g]=th
        if not (np.isnan(pvg[3]) or np.isnan(pvg[0])):
            perm_sp.append(pvg[3]-pvg[0])
    
    perm_arr = np.array(perm_sp)
    # Phipson & Smyth (2010) bias-corrected p-value
    if len(perm_arr)>0 and not np.isnan(obs_sp):
        count = np.sum(np.abs(perm_arr) >= abs(obs_sp))
        pval = (1 + count) / (1 + len(perm_arr))
    else:
        pval = np.nan
    
    return gv, gs, cate_B, q_groups, obs_sp, perm_arr, float(pval)

all_res = []
fig, axes = plt.subplots(2,3,figsize=(17,10))
fig.suptitle("Step 7 v4 — Honest GATES + 200 Permutations (Phipson–Smyth p-value)\n"
             "CausalForestDML on half-A; CATE quartiles from held-out half-B predictions\n"
             "Features: brand, UPC, week seasonality — NO depth/price (pre-treatment only)",
             fontsize=10, fontweight='bold')
cc = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}

for col_i,cat in enumerate(CATS):
    print(f"\n{'='*45}\n  {cat.upper()}")
    X,Y,W = load_and_build(cat,1,2000)
    print(f"  n={len(Y)}, W=1 rate={W.mean():.3f}")
    
    gv,gs,cate_B,qg,obs_sp,perm_arr,pval = honest_gates_cfml(X,Y,W,n_perm=200)
    
    for g in range(4): print(f"    Q{g+1}: GATE={gv[g]:.4f} SE={gs[g]:.4f}" if not np.isnan(gv[g]) else f"    Q{g+1}: insufficient")
    p_min = 1/201  # smallest possible with 200 perms
    print(f"  CATE quartile spread: {obs_sp:.4f}")
    print(f"  Perm p-val (Phipson-Smyth): {pval:.4f}  [min possible = {p_min:.4f} = 1/201]")
    print(f"  Mean pred CATE: {np.nanmean(cate_B):.4f}  SD: {np.nanstd(cate_B):.4f}")
    
    sig = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else "n.s."))
    all_res.append({"category":cat,"n_obs":len(Y),
                    "mean_cate":round(np.nanmean(cate_B),4),
                    "sd_cate":round(np.nanstd(cate_B),4),
                    "gate_q1":round(gv[0],4),"se_q1":round(gs[0],4) if not np.isnan(gs[0]) else np.nan,
                    "gate_q2":round(gv[1],4) if not np.isnan(gv[1]) else np.nan,
                    "gate_q3":round(gv[2],4) if not np.isnan(gv[2]) else np.nan,
                    "gate_q4":round(gv[3],4),"se_q4":round(gs[3],4) if not np.isnan(gs[3]) else np.nan,
                    "obs_spread":round(obs_sp,4),"n_perm":len(perm_arr),
                    "pval_corrected":round(pval,4),"sig":sig,
                    "pval_note":"Phipson-Smyth (1+count)/(1+200); min p=0.005"})
    
    ax_top = axes[0,col_i]
    ax_bot = axes[1,col_i]
    col = cc[cat]
    
    # Top: GATE bars
    qnames=["Q1","Q2","Q3","Q4"]; gvals=[gv[g] for g in range(4)]; gserr=[gs[g] for g in range(4)]
    alph=[0.35,0.55,0.75,1.0]
    for qn,gval,gse,a in zip(qnames,gvals,gserr,alph):
        if np.isnan(gval): continue
        ax_top.bar(qn,gval,color=col,alpha=a,edgecolor='black',lw=1.2,zorder=3)
        if not np.isnan(gse): ax_top.errorbar(qn,gval,yerr=1.96*gse,fmt='none',color='black',capsize=5,lw=2,zorder=4)
    ax_top.axhline(0,color='gray',lw=1,ls='--')
    ax_top.set_title(f"{cat.capitalize()}\nQ4−Q1={obs_sp:.3f}  p={pval:.3f} ({sig})",fontsize=11)
    ax_top.set_xlabel("CATE Quartile Group\n(Q1=lowest pred. CATE, Q4=highest)")
    ax_top.set_ylabel("GATE (log-units)"); ax_top.grid(True,alpha=0.3,axis='y')
    
    # Bottom: CATE histogram + permutation null
    ax_bot.hist(cate_B,bins=25,color=col,alpha=0.6,density=True,label="Pred. CATE dist.")
    ax_bot.set_title(f"Predicted CATE distribution\nMean={np.nanmean(cate_B):.3f}  SD={np.nanstd(cate_B):.3f}",fontsize=10)
    ax_bot.set_xlabel("Predicted CATE")
    ax_bot.set_ylabel("Density")
    axi = ax_bot.inset_axes([0.60,0.45,0.38,0.50])
    axi.hist(perm_arr,bins=30,color='gray',alpha=0.7,density=True,zorder=2)
    axi.axvline(obs_sp,color='red',lw=2)
    axi.set_title(f"Perm null\np={pval:.3f}",fontsize=7); axi.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step07_honest_gates_v4.png",dpi=150,bbox_inches='tight')
plt.close()

out=pd.DataFrame(all_res)
out.to_csv(f"{OUT_DIR}/honest_gates_v4_cfml.csv",index=False)
print("\n=== GATES SUMMARY ===")
print(out[["category","mean_cate","gate_q1","gate_q4","obs_spread","pval_corrected","sig"]].to_string(index=False))
print("\nFigure saved.")
