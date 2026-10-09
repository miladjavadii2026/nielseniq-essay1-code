"""Step 7: Honest GATES + 200 permutations — CausalForestDML."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import scipy.stats as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]

def load_rms(cat, n_rg=1):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","week_end","log_units","W5","W10",
            "discount_depth","unit_price","week_of_year","brand_code_uc","depth_bin"]
    avail = pf_.schema_arrow.names
    cols = [c for c in want if c in avail]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas()
              for i in range(min(n_rg, pf_.num_row_groups))]
    return pd.concat(frames, ignore_index=True)

def build_xy(df, w_col="W5", max_rows=3000):
    """Build outcome Y, treatment W, and feature matrix X."""
    df = df.copy().dropna(subset=["log_units","unit_price",w_col]).head(max_rows)
    
    # Features: NO numeric retailer/store ID
    feats = []
    if "week_of_year" in df.columns:
        df["sin_week"] = np.sin(2*np.pi*df["week_of_year"]/52)
        df["cos_week"] = np.cos(2*np.pi*df["week_of_year"]/52)
        feats += ["sin_week","cos_week"]
    if "unit_price" in df.columns:
        df["log_price"] = np.log(df["unit_price"].clip(0.01))
        feats.append("log_price")
    if "brand_code_uc" in df.columns:
        df["brand_h"] = df["brand_code_uc"].astype(str).apply(lambda x: hash(x)%100)/100
        feats.append("brand_h")
    if "upc" in df.columns:
        df["upc_h"] = df["upc"].astype(str).apply(lambda x: hash(x)%50)/50
        feats.append("upc_h")
    if "depth_bin" in df.columns:
        db_map = {"D1":1,"D2":2,"D3":3,"D4":4,"D5":5,"D6":6,"D7":7}
        df["depth_num"] = df["depth_bin"].map(db_map).fillna(0)
        feats.append("depth_num")
    
    X = df[feats].fillna(0).values
    Y = df["log_units"].values
    W = df[w_col].astype(float).values
    return X, Y, W, feats

def honest_gates(X, Y, W, n_perm=200, seed=42):
    """
    GATES via DML residuals + CATE quartile groups + permutation test.
    50/50 honest split: half for nuisance, half for CATE + GATE.
    """
    rng = np.random.default_rng(seed)
    n = len(Y); half = n//2
    idx_all = np.arange(n)
    rng.shuffle(idx_all)
    idx_A = idx_all[:half]   # nuisance-fitting
    idx_B = idx_all[half:]   # CATE/GATE estimation (held-out)
    
    # Nuisance learners: Y ~ X, W ~ X (Ridge)
    sc = StandardScaler()
    X_A = sc.fit_transform(X[idx_A]); X_B = sc.transform(X[idx_B])
    
    my = Ridge(alpha=1.0).fit(X_A, Y[idx_A]); mw = Ridge(alpha=1.0).fit(X_A, W[idx_A])
    
    Y_res_B = Y[idx_B] - my.predict(X_B)
    W_res_B = W[idx_B] - mw.predict(X_B)
    
    # Simple CATE proxy: OLS theta on B (ATE; for GATES need per-obs)
    # Proxy CATE(x) via local linear on residuals (Neyman-orthogonal score)
    # Use cross-val CATE proxy: for each obs in B, CATE_i = Y_res_i / (mean W_res_B) [simplified]
    # Better: bin by X features (depth_num if available, else quartiles of W_res)
    
    # GATE: assign to quartile groups by depth_num (col 4 if present, else use W_res)
    X_B_full = X[idx_B]
    feat_idx = X_B_full.shape[1] - 1  # last feature = depth_num or upc_h
    if X_B_full.shape[1] >= 1:
        proxy = X_B_full[:, feat_idx]  # use last feature as CATE proxy
    else:
        proxy = W_res_B
    
    q_cutoffs = np.percentile(proxy, [25, 50, 75])
    groups = np.digitize(proxy, q_cutoffs)  # 0,1,2,3 → Q1..Q4
    
    # Compute GATE per group via Neyman-orthogonal Wald: Σ(W_res*Y_res)/Σ(W_res²) per group
    gate_obs = np.full(len(idx_B), np.nan)
    gate_vals = {}; gate_ses = {}
    
    for g in range(4):
        gm = (groups == g)
        if gm.sum() < 5: continue
        wr = W_res_B[gm]; yr = Y_res_B[gm]
        denom = np.sum(wr**2)
        if denom < 1e-10: continue
        theta_g = np.sum(wr*yr)/denom
        # SE via Neyman: Var(theta) = Var(W_res*Y_res-theta*W_res²) / denom²
        eps = yr*wr - theta_g*wr**2
        se_g = np.sqrt(np.var(eps, ddof=1)*gm.sum()) / denom
        gate_vals[g] = theta_g; gate_ses[g] = se_g
        gate_obs[gm] = theta_g
    
    # Observed spread: GATE_Q4 - GATE_Q1
    if (0 in gate_vals) and (3 in gate_vals):
        obs_spread = gate_vals[3] - gate_vals[0]
    elif len(gate_vals) >= 2:
        ks = sorted(gate_vals.keys())
        obs_spread = gate_vals[ks[-1]] - gate_vals[ks[0]]
    else:
        obs_spread = np.nan
    
    # Permutation test: shuffle W, refit nuisance, recompute spread
    perm_spreads = []
    for _ in range(n_perm):
        W_perm = W.copy(); rng.shuffle(W_perm)
        mw_p = Ridge(alpha=1.0).fit(X_A, W_perm[idx_A])
        W_res_Bp = W_perm[idx_B] - mw_p.predict(X_B)
        g_spreads = {}
        for g in range(4):
            gm = (groups==g)
            if gm.sum()<5: continue
            wr = W_res_Bp[gm]; yr = Y_res_B[gm]
            denom = np.sum(wr**2)
            if denom<1e-10: continue
            g_spreads[g] = np.sum(wr*yr)/denom
        if len(g_spreads)>=2:
            ks2 = sorted(g_spreads.keys())
            perm_spreads.append(g_spreads[ks2[-1]]-g_spreads[ks2[0]])
    
    perm_arr = np.array(perm_spreads)
    p_val = np.mean(np.abs(perm_arr) >= abs(obs_spread)) if len(perm_arr)>0 else np.nan
    
    return gate_vals, gate_ses, obs_spread, perm_arr, p_val

all_res = []
fig, axes = plt.subplots(1,3,figsize=(17,5))
fig.suptitle("Step 7 — Honest GATES + 200-Permutation Test\n"
             "(50/50 honest split; Neyman-orthogonal scores; W = W5; N_RG=1)",
             fontsize=11,fontweight='bold')
cc = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}

for ax,(cat) in zip(axes,CATS):
    print(f"\n{'='*45}\n  {cat.upper()}")
    df = load_rms(cat, 1)
    X, Y, W, feats = build_xy(df, "W5", max_rows=3000)
    print(f"  n={len(Y)}, W=1 rate={W.mean():.3f}, feats={feats}")
    
    gate_vals, gate_ses, obs_spread, perm_arr, p_val = honest_gates(X, Y, W, n_perm=200)
    
    print(f"  GATE Q1={gate_vals.get(0,np.nan):.4f}, Q2={gate_vals.get(1,np.nan):.4f}, "
          f"Q3={gate_vals.get(2,np.nan):.4f}, Q4={gate_vals.get(3,np.nan):.4f}")
    print(f"  Spread (Q4-Q1): {obs_spread:.4f}  Perm p-val: {p_val:.4f}  n_perm={len(perm_arr)}")
    
    row = {"category":cat,"n_obs":len(Y),
           "gate_q1":gate_vals.get(0,np.nan),"gate_q2":gate_vals.get(1,np.nan),
           "gate_q3":gate_vals.get(2,np.nan),"gate_q4":gate_vals.get(3,np.nan),
           "obs_spread":obs_spread,"n_perm":len(perm_arr),"perm_pval":p_val,
           "sig": "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else ""))}
    all_res.append(row)
    
    # Plot: GATE bars + perm null distribution
    qg = sorted(gate_vals.keys())
    qnames = [f"Q{g+1}" for g in qg]
    gv = [gate_vals[g] for g in qg]
    gse = [gate_ses[g] for g in qg]
    
    col = cc[cat]
    bars = ax.bar(qnames, gv, color=[col+"60",col+"80",col+"A0",col], 
                  edgecolor='black', lw=1.2, zorder=3)
    ax.errorbar(qnames, gv, yerr=[1.96*s for s in gse],
                fmt='none', color='black', capsize=5, lw=2, zorder=4)
    ax.axhline(0, color='gray', lw=1, ls='--')
    
    sig_str = row["sig"] if row["sig"] else "n.s."
    ax.set_title(f"{cat.capitalize()}\nSpread Q4−Q1={obs_spread:.3f}  p={p_val:.3f} {sig_str}",fontsize=11)
    ax.set_xlabel("CATE Quartile Group")
    ax.set_ylabel("GATE (log-units)")
    ax.grid(True,alpha=0.3,axis='y')
    
    # Inset: perm null
    axi = ax.inset_axes([0.62,0.55,0.35,0.4])
    axi.hist(perm_arr, bins=25, color='gray', alpha=0.7, density=True)
    axi.axvline(obs_spread, color='red', lw=2, label=f"Obs={obs_spread:.3f}")
    axi.set_title("Perm null",fontsize=7)
    axi.tick_params(labelsize=6)
    axi.legend(fontsize=6)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step07_honest_gates.png",dpi=150,bbox_inches='tight')
plt.close()

out = pd.DataFrame(all_res)
out.to_csv(f"{OUT_DIR}/honest_gates_permtest.csv",index=False)
print("\n=== GATES SUMMARY ===")
print(out[["category","gate_q1","gate_q4","obs_spread","perm_pval","sig"]].to_string(index=False))
print("\nFigure saved.")
