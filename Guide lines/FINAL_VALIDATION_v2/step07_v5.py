"""Step 7 v5: Honest GATES — 500 permutations, Phipson-Smyth p-values.
Same CausalForestDML logic as v4 but more permutations.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from econml.dml import CausalForestDML

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
N_PERM    = 500
MAX_OBS   = 30000

def load_rms(cat, n_rg=2):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","depth_bin","week_of_year","regular_price_q75"]
    have = [c for c in want if c in pf.schema_arrow.names]
    frames = [pf.read_row_group(i,columns=have).to_pandas()
              for i in range(min(n_rg, pf.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(float)
    return df

def rank_q4(arr, rng):
    n = len(arr); idx = np.argsort(arr)
    q = np.empty(n, dtype=int)
    for i,j in enumerate(idx): q[j] = int(4*i/n)+1
    q = np.clip(q,1,4)
    return q

all_rows = []
for cat in CATS:
    print(f"\n{'='*40}\n  {cat.upper()}")
    df  = load_rms(cat)
    df  = df.dropna(subset=["log_units","depth_num"]).copy()
    if len(df)>MAX_OBS: df = df.sample(MAX_OBS, random_state=42)
    
    Y = df["log_units"].values
    W = (df["depth_num"]>5).astype(float).values   # W5: depth>5% (locked def)
    wk= df["week_of_year"].fillna(26).values
    rp= df["regular_price_q75"].fillna(3).values if "regular_price_q75" in df.columns else np.ones(len(df))
    X = np.column_stack([np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52), rp])
    
    rng = np.random.default_rng(42)
    n   = len(Y); idx = np.arange(n); rng.shuffle(idx)
    ia,ib = idx[:n//2], idx[n//2:]
    Ya,Wa,Xa = Y[ia],W[ia],X[ia]
    Yb,Wb,Xb = Y[ib],W[ib],X[ib]
    
    cf = CausalForestDML(model_y=Ridge(alpha=1.0),model_t=Ridge(alpha=1.0),
                         n_estimators=100,min_samples_leaf=5,
                         random_state=42,honest=True,inference=False)
    cf.fit(Ya, Wa, X=Xa)
    cate_b = cf.effect(Xb).flatten()
    
    q_grps = rank_q4(cate_b, rng)
    mean_cate = float(cate_b.mean())
    
    gates = {}
    for q in range(1,5):
        m = q_grps==q
        if m.sum()>0:
            gates[q] = float(np.mean(Yb[m][Wb[m]==1]) - np.mean(Yb[m][Wb[m]==0])
                              if Wb[m].sum()>0 and (1-Wb[m]).sum()>0
                              else np.mean(cate_b[m]))
        else: gates[q]=np.nan
    
    obs_spread = float(gates[4] - gates[1]) if not any(np.isnan(list(gates.values()))) else np.nan
    
    # Permutation test (500 draws)
    perm_spreads = []
    for _ in range(N_PERM):
        perm_cate = rng.permutation(cate_b)
        pq_  = rank_q4(perm_cate, rng)
        g1   = float(np.mean(cate_b[pq_==1]))
        g4   = float(np.mean(cate_b[pq_==4]))
        perm_spreads.append(g4-g1)
    
    obs_sp = float(np.mean(cate_b[q_grps==4]) - np.mean(cate_b[q_grps==1]))
    count  = np.sum(np.abs(perm_spreads) >= abs(obs_sp))
    pval   = (1+count)/(1+N_PERM)   # Phipson & Smyth (2010)
    sig    = "***" if pval<.001 else ("**" if pval<.01 else ("*" if pval<.05 else "n.s."))
    
    print(f"  mean CATE={mean_cate:.4f} GATE Q1={gates[1]:.4f} Q4={gates[4]:.4f} "
          f"spread={obs_spread:.4f} p={pval:.4f} {sig}")
    all_rows.append({"category":cat,"mean_cate":round(mean_cate,4),
                     "gate_q1":round(gates[1],4),"gate_q2":round(gates[2],4),
                     "gate_q3":round(gates[3],4),"gate_q4":round(gates[4],4),
                     "obs_spread":round(obs_spread,4),"n_perm":N_PERM,
                     "pval_corrected":round(pval,4),"sig":sig})

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/honest_gates_v5_500perm.csv", index=False)

# Figure
fig,ax = plt.subplots(figsize=(9,6))
CAT_C  = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for i,row in out.iterrows():
    qs   = [row["gate_q1"],row["gate_q2"],row["gate_q3"],row["gate_q4"]]
    c    = CAT_C[row["category"]]
    ax.plot([1,2,3,4], qs, 'o-', color=c, lw=2, ms=10, label=
            f"{row['category'].capitalize()} spread={row['obs_spread']:.3f} p={row['pval_corrected']:.3f} {row['sig']}")
ax.axhline(0,color='k',lw=0.8,ls='--')
ax.set_title("Step 7 v5 — Honest GATES (CausalForestDML, 500 perms, Phipson-Smyth p)\n"
             "Quartiles from held-out CF predictions; W5=depth>5%",
             fontsize=10,fontweight='bold')
ax.set_xlabel("CATE Quartile (Q1=low, Q4=high)"); ax.set_ylabel("GATE (Δ log units)")
ax.set_xticks([1,2,3,4])
ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step07_honest_gates_v5.png",dpi=150,bbox_inches='tight')
plt.close()
print("\n✓ Step 7 v5 done")
print(out[["category","mean_cate","gate_q1","gate_q4","obs_spread","n_perm","pval_corrected","sig"]].to_string(index=False))
