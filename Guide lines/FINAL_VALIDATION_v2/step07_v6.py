"""Step 7 v6: Honest GATES using CF-prediction means (not raw T−C diff).
Fix: GATE[q] = mean(cate_b[q_grp==q]) — the honest CF estimate in each quartile.
Raw T−C differences within quartile are selection-biased and NOT GATE estimates.
Also: propensity overlap trim (0.1 < P(W=1|X) < 0.9); winsorized sensitivity.
500 permutations, Phipson-Smyth p-values.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge, LogisticRegression
from econml.dml import CausalForestDML

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
N_PERM    = 500
MAX_OBS   = 30000
PROP_LO   = 0.10   # overlap trim
PROP_HI   = 0.90

def load_rms(cat):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["log_units","depth_bin","week_of_year","regular_price_q75"]
    have = [c for c in want if c in pf.schema_arrow.names]
    frames = [pf.read_row_group(i, columns=have).to_pandas()
              for i in range(min(2, pf.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(float)
    return df

all_rows = []
for cat in CATS:
    print(f"\n{'='*50}\n  {cat.upper()}")
    df  = load_rms(cat).dropna(subset=["log_units","depth_num"])
    if len(df)>MAX_OBS: df = df.sample(MAX_OBS, random_state=42)
    
    Y = df["log_units"].values
    W = (df["depth_num"]>5).astype(float).values   # W5: strictly >5%
    wk= df["week_of_year"].fillna(26).values.astype(float)
    rp= df["regular_price_q75"].fillna(3).values if "regular_price_q75" in df.columns else np.ones(len(df))
    X = np.column_stack([np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52), rp])

    # Overlap trim: estimate propensity, exclude extreme obs
    try:
        lr = LogisticRegression(max_iter=300, C=1.0)
        lr.fit(X, W)
        prop = lr.predict_proba(X)[:,1]
        keep = (prop>PROP_LO) & (prop<PROP_HI)
        n_trim = (~keep).sum()
        print(f"  Overlap trim: removed {n_trim}/{len(df)} obs (prop outside [{PROP_LO},{PROP_HI}])")
        if keep.sum() < 200:
            print("  *** Too few after trim, skip overlap trim"); keep = np.ones(len(df), dtype=bool)
        Y,W,X = Y[keep], W[keep], X[keep]
    except Exception as e:
        print(f"  Propensity trim failed: {e}"); n_trim=0

    rng = np.random.default_rng(42)
    n   = len(Y); idx = np.arange(n); rng.shuffle(idx)
    ia, ib = idx[:n//2], idx[n//2:]
    Ya,Wa,Xa = Y[ia],W[ia],X[ia]
    Yb,Wb,Xb = Y[ib],W[ib],X[ib]

    cf = CausalForestDML(model_y=Ridge(alpha=1.0), model_t=Ridge(alpha=1.0),
                         n_estimators=100, min_samples_leaf=5,
                         random_state=42, honest=True, inference=False)
    cf.fit(Ya, Wa, X=Xa)
    cate_b = cf.effect(Xb).flatten()

    # CORRECT GATE: mean of CF predictions by quartile (NOT raw T−C)
    def rank_q4(arr):
        n = len(arr); idx2 = np.argsort(arr)
        q = np.empty(n, dtype=int)
        for i,j in enumerate(idx2): q[j] = int(4*i/n)+1
        return np.clip(q, 1, 4)

    q_grps = rank_q4(cate_b)
    gates  = {qi: float(np.mean(cate_b[q_grps==qi])) for qi in range(1,5)}
    mean_cate = float(cate_b.mean())

    # Winsorised sensitivity (99th pct clip)
    cate_w = np.clip(cate_b, np.percentile(cate_b,1), np.percentile(cate_b,99))
    qw = rank_q4(cate_w)
    gates_w = {qi: float(np.mean(cate_w[qw==qi])) for qi in range(1,5)}

    obs_sp = gates[4] - gates[1]

    # Permutation test on CF predictions
    perm_spreads = []
    for _ in range(N_PERM):
        pc = rng.permutation(cate_b)
        pq_ = rank_q4(pc)
        perm_spreads.append(float(np.mean(pc[pq_==4]) - np.mean(pc[pq_==1])))
    count = np.sum(np.abs(perm_spreads) >= abs(obs_sp))
    pval  = (1+count)/(1+N_PERM)
    sig   = "***" if pval<.001 else ("**" if pval<.01 else ("*" if pval<.05 else "n.s."))

    print(f"  mean CATE={mean_cate:.4f}  GATE Q1={gates[1]:.4f} Q4={gates[4]:.4f}  "
          f"spread={obs_sp:.4f}  p={pval:.4f} {sig}")
    print(f"  [Winsorized] Q1={gates_w[1]:.4f} Q4={gates_w[4]:.4f} spread={gates_w[4]-gates_w[1]:.4f}")
    all_rows.append({"category":cat,"mean_cate":round(mean_cate,4),
                     "gate_q1":round(gates[1],4),"gate_q2":round(gates[2],4),
                     "gate_q3":round(gates[3],4),"gate_q4":round(gates[4],4),
                     "obs_spread":round(obs_sp,4),
                     "gate_q1_wins":round(gates_w[1],4),"gate_q4_wins":round(gates_w[4],4),
                     "n_perm":N_PERM,"pval":round(pval,4),"sig":sig,"n_trim":n_trim})

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/honest_gates_v6_corrected.csv", index=False)

fig,ax=plt.subplots(figsize=(9,6))
CC={"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for i,row in out.iterrows():
    qs=[row["gate_q1"],row["gate_q2"],row["gate_q3"],row["gate_q4"]]
    ax.plot([1,2,3,4],qs,'o-',color=CC[row["category"]],lw=2,ms=10,
            label=f"{row['category'].capitalize()} spread={row['obs_spread']:.3f} p={row['pval']:.3f} {row['sig']}")
ax.axhline(0,color='k',lw=0.8,ls='--')
ax.set_title("Step 7 v6 — Honest GATES (CF prediction means, overlap-trimmed)\n"
             "GATE[q] = mean(CF predictions in quartile); NOT raw T−C difference; 500 perms",
             fontsize=10,fontweight='bold')
ax.set_xlabel("CATE Quartile"); ax.set_ylabel("GATE (mean CF prediction, Δ log units)")
ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step07_honest_gates_v6.png",dpi=150,bbox_inches='tight')
plt.close()
print("\n✓ Step 7 v6 done")
print(out[["category","mean_cate","gate_q1","gate_q4","obs_spread","pval","sig","n_trim"]].to_string(index=False))
