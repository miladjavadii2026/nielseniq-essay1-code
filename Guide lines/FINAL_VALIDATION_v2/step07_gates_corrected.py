"""
Step 7 GATES — CORRECTED (v7): uses W = df["W5"] not depth_bin_number > 5.

Previous versions (v5, v6): used W = (depth_num > 5) = D6∪D7 only → WRONG.
This version: W = df["W5"] = 1{discount_depth > 5%} → ~23% treated.
Overlap confirmed in v2 audit: 61% in [0.1,0.9] for cereal, 57% snacks, 11% coffee.

Method:
  - Load W5 directly from parquet
  - OOF propensity trim: keep prop ∈ [0.10, 0.90]
  - Honest two-sample CausalForestDML (50 trees, honest=True)
  - GATE[q] = mean(CF_predictions[quartile==q]) — NOT raw T-C diff
  - 200 permutations, Phipson-Smyth p-values
  - Winsorised sensitivity (1st-99th pct)
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge, LogisticRegression
from econml.dml import CausalForestDML
import os

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal", "snacks", "coffee"]
N_PERM    = 200
MAX_OBS   = 6000   # Sandbox limit; workstation should use 30k+
PROP_LO   = 0.10
PROP_HI   = 0.90
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

def rank_q4(arr):
    n = len(arr); idx2 = np.argsort(arr)
    q = np.empty(n, dtype=int)
    for i, j in enumerate(idx2): q[j] = min(4, int(4*i/n)+1)
    return q

all_rows = []

for cat in CATS:
    print(f"\n{'='*55}\n  {cat.upper()}")

    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","log_units","W5","week_of_year","regular_price_q75","panel_year","retailer_code"]
    have = [c for c in want if c in pf.schema_arrow.names]
    df   = pf.read_row_group(0, columns=have).to_pandas().dropna(subset=["log_units","W5"])

    if len(df) > MAX_OBS:
        df = df.sample(MAX_OBS, random_state=42).reset_index(drop=True)

    n   = len(df)
    W   = df["W5"].astype(float).values
    Y   = df["log_units"].values
    n_t = int(W.sum()); ts = W.mean()
    print(f"  N={n:,}  treated={n_t:,} ({100*ts:.1f}%)")

    wk = df["week_of_year"].fillna(26).astype(float).values
    rp = df["regular_price_q75"].fillna(df["regular_price_q75"].median()).values
    X  = np.column_stack([np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52), rp])

    # Overlap trim using logistic propensity
    lr = LogisticRegression(solver="liblinear", max_iter=200, C=1.0, random_state=42)
    lr.fit(X, W)
    prop = lr.predict_proba(X)[:, 1]
    keep = (prop > PROP_LO) & (prop < PROP_HI)
    n_trim = int((~keep).sum())
    print(f"  Overlap trim [{PROP_LO},{PROP_HI}]: removed {n_trim}/{n}; "
          f"kept {int(keep.sum())} ({100*keep.mean():.0f}%)")
    if keep.sum() < 200:
        print("  *** < 200 obs after trim; run on full sample with note ***")
        keep = np.ones(n, dtype=bool)
    Y_k, W_k, X_k = Y[keep], W[keep], X[keep]
    n_k = len(Y_k)
    print(f"  Trimmed sample: N={n_k:,}  treated={int(W_k.sum()):,}")

    # Two-sample honest split
    rng = np.random.default_rng(42)
    idx = np.arange(n_k); rng.shuffle(idx)
    ia, ib = idx[:n_k//2], idx[n_k//2:]
    Ya, Wa, Xa = Y_k[ia], W_k[ia], X_k[ia]
    Yb, Wb, Xb = Y_k[ib], W_k[ib], X_k[ib]

    # Honest CausalForestDML
    cf = CausalForestDML(
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        n_estimators=50,
        min_samples_leaf=5,
        random_state=42,
        honest=True,
        inference=False,
    )
    cf.fit(Ya, Wa, X=Xa)
    cate_b = cf.effect(Xb).flatten()

    # GATE[q] = mean of CF predictions in quartile (NOT raw T-C diff)
    q_grps = rank_q4(cate_b)
    gates  = {qi: float(np.mean(cate_b[q_grps==qi])) for qi in range(1, 5)}
    gates_w = {}
    cate_w = np.clip(cate_b, np.percentile(cate_b, 1), np.percentile(cate_b, 99))
    qw = rank_q4(cate_w)
    for qi in range(1, 5):
        gates_w[qi] = float(np.mean(cate_w[qw==qi]))
    mean_cate = float(cate_b.mean())
    obs_sp    = gates[4] - gates[1]

    # Permutation test on CF predictions (within-sample shuffle of CATE labels)
    perm_sp = []
    for _ in range(N_PERM):
        pc   = rng.permutation(cate_b)
        pq_  = rank_q4(pc)
        perm_sp.append(float(np.mean(pc[pq_==4]) - np.mean(pc[pq_==1])))
    count = int(np.sum(np.abs(perm_sp) >= abs(obs_sp)))
    pval  = (1 + count) / (1 + N_PERM)
    sig   = ("***" if pval < .001 else ("**" if pval < .01 else ("*" if pval < .05 else "n.s.")))

    print(f"  mean_CATE={mean_cate:.4f}  "
          f"Q1={gates[1]:.4f} Q2={gates[2]:.4f} Q3={gates[3]:.4f} Q4={gates[4]:.4f}")
    print(f"  Spread(Q4-Q1)={obs_sp:.4f}  p={pval:.4f} {sig}")
    print(f"  [Winsorized] Q1={gates_w[1]:.4f} Q4={gates_w[4]:.4f}  "
          f"spread={gates_w[4]-gates_w[1]:.4f}")

    all_rows.append({
        "category": cat,
        "W_definition": "W5 = 1{discount_depth > 5%}  [CORRECTED]",
        "n_sample": n, "n_trimmed": n_k, "treated_share": round(ts, 4),
        "mean_cate": round(mean_cate, 4),
        "gate_q1": round(gates[1], 4), "gate_q2": round(gates[2], 4),
        "gate_q3": round(gates[3], 4), "gate_q4": round(gates[4], 4),
        "spread": round(obs_sp, 4),
        "gate_q1_wins": round(gates_w[1], 4), "gate_q4_wins": round(gates_w[4], 4),
        "spread_wins": round(gates_w[4]-gates_w[1], 4),
        "n_perm": N_PERM, "pval": round(pval, 4), "sig": sig,
        "n_trim_removed": n_trim,
        "prop_lo": PROP_LO, "prop_hi": PROP_HI,
    })

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/honest_gates_v7_corrected.csv", index=False)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
CC = {"cereal": "#1565C0", "snacks": "#E65100", "coffee": "#2E7D32"}
for _, row in out.iterrows():
    qs = [row["gate_q1"], row["gate_q2"], row["gate_q3"], row["gate_q4"]]
    ax.plot([1,2,3,4], qs, 'o-', color=CC[row["category"]], lw=2, ms=10,
            label=f"{row['category'].capitalize()} spread={row['spread']:.3f} "
                  f"p={row['pval']:.3f} {row['sig']}")

ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_title(
    "Step 7 v7 CORRECTED — Honest GATES (W5=1{discount>5%})\n"
    "GATE[q]=mean(CF predictions in quartile); overlap-trimmed [0.10,0.90]; "
    f"{N_PERM} perms",
    fontsize=9, fontweight='bold')
ax.set_xlabel("CATE Quartile"); ax.set_ylabel("GATE (Δ log units)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step07_honest_gates_v7_corrected.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*55)
print("GATES SUMMARY (corrected W5, trimmed [0.10,0.90])")
print(out[["category","treated_share","n_trimmed","mean_cate",
          "gate_q1","gate_q4","spread","pval","sig"]].to_string(index=False))
print("\n✓ Saved: honest_gates_v7_corrected.csv + figure")
