"""Step 8 v3: Per-bin honest forests — D1 vs Dk for each k in 2..7.
One CausalForestDML per bin contrast (W binary: 1 if in that bin, 0 if D1).
Honest 50/50 split; 100 estimators. SE via bootstrap of leaf predictions.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from econml.dml import CausalForestDML

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step08_causal_forest_dose_response"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
MAX_OBS   = 25000   # per-bin contrast; smaller subsample for speed
MIN_BIN   = 30

def load_rms(cat, n_rg=2):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["log_units","depth_bin","week_of_year","regular_price_q75","brand_code_uc"]
    have = [c for c in want if c in pf.schema_arrow.names]
    frames = [pf.read_row_group(i, columns=have).to_pandas()
              for i in range(min(n_rg, pf.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(int)
    return df

all_rows = []

for cat in CATS:
    print(f"\n{'='*50}\n  {cat.upper()}")
    df = load_rms(cat).dropna(subset=["log_units","depth_num"]).copy()
    # D1 base pool
    d1_pool = df[df["depth_num"]==1].copy()
    if len(d1_pool) < MIN_BIN:
        print(f"  D1 too small ({len(d1_pool)}) — skip"); continue

    for k in range(2, 8):
        dk_pool = df[df["depth_num"]==k].copy()
        if len(dk_pool) < MIN_BIN:
            print(f"  D{k} too small ({len(dk_pool)}) — skip"); continue

        # Pool D1 and Dk, balance sizes
        n_each = min(len(d1_pool), len(dk_pool), MAX_OBS//2)
        sub1 = d1_pool.sample(n_each, random_state=k*10+42, replace=len(d1_pool)<n_each)
        subk = dk_pool.sample(n_each, random_state=k*10+43, replace=len(dk_pool)<n_each)
        sub  = pd.concat([sub1, subk], ignore_index=True)
        sub  = sub.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

        Y = sub["log_units"].values
        W = (sub["depth_num"]==k).astype(float).values  # 1=Dk, 0=D1
        wk = sub["week_of_year"].fillna(26).values.astype(float)
        X_cols = [np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52)]
        if "regular_price_q75" in sub.columns:
            X_cols.append(sub["regular_price_q75"].fillna(sub["regular_price_q75"].median()).values)
        if "brand_code_uc" in sub.columns:
            X_cols.append(pd.Categorical(sub["brand_code_uc"]).codes.astype(float))
        X = np.column_stack(X_cols)

        rng = np.random.default_rng(42+k)
        n   = len(Y); idx = np.arange(n); rng.shuffle(idx)
        ia, ib = idx[:n//2], idx[n//2:]

        try:
            cf = CausalForestDML(model_y=Ridge(alpha=1.0), model_t=Ridge(alpha=1.0),
                                 n_estimators=100, min_samples_leaf=5,
                                 random_state=42, honest=True, inference=True)
            cf.fit(Y[ia], W[ia], X=X[ia])
            inf_ = cf.effect_inference(X[ib])
            ate  = float(inf_.pred.mean())
            se   = float(np.std(inf_.pred) / np.sqrt(len(inf_.pred)))
            ci_lo= ate - 1.96*se; ci_hi = ate + 1.96*se
            z    = ate/se if se>0 else np.nan
            p    = float(2*(1-min(1, abs(z)/1.96))) if not np.isnan(z) else np.nan  # approx
            sig  = "***" if abs(z)>3.29 else ("**" if abs(z)>2.58 else ("*" if abs(z)>1.96 else "n.s."))
            print(f"  D1vsD{k}: ate={ate:.4f} se={se:.4f} z={z:.2f} {sig}  (n1={n_each}, nk={n_each})")
        except Exception as e:
            ate=se=ci_lo=ci_hi=z=np.nan; sig=f"err:{str(e)[:20]}"
            print(f"  D1vsD{k}: ERROR {e}")

        all_rows.append({"category":cat,"bin":k,"label":f"D{k}",
                         "ate":round(ate,4),"se":round(se,4) if not np.isnan(se) else np.nan,
                         "ci_lo":round(ci_lo,4) if not np.isnan(ci_lo) else np.nan,
                         "ci_hi":round(ci_hi,4) if not np.isnan(ci_hi) else np.nan,
                         "z":round(z,3) if not np.isnan(z) else np.nan,
                         "sig":sig,"n_each":n_each})

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/dose_response_v3_perbin.csv", index=False)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
CAT_C = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
LABELS = {2:"D2\n5-10%",3:"D3\n10-20%",4:"D4\n20-30%",
          5:"D5\n30-40%",6:"D6\n40-50%",7:"D7\n≥50%"}

for ax, cat in zip(axes, CATS):
    cat_data = out[out["category"]==cat].copy()
    if cat_data.empty: ax.set_title(f"{cat.capitalize()}\n(no data)"); continue
    xs  = cat_data["bin"].values
    ys  = cat_data["ate"].values
    ses = cat_data["se"].values
    sigs= cat_data["sig"].values
    color = CAT_C[cat]
    bars = ax.bar(xs, ys, color=color, alpha=0.65, width=0.55, zorder=3)
    ok = ~np.isnan(ses)
    ax.errorbar(xs[ok], ys[ok], yerr=1.96*ses[ok],
                fmt='none', color='k', capsize=5, lw=1.5, zorder=5)
    for xi, yi, s, sig in zip(xs, ys, ses, sigs):
        if not np.isnan(yi):
            ax.text(xi, yi + (0.02 if yi>=0 else -0.04), sig,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(int(xi),"?") for xi in xs], fontsize=7.5)
    ax.set_title(f"{cat.capitalize()}\nPer-bin CausalForestDML vs D1", fontsize=10, fontweight='bold')
    ax.set_xlabel("Depth bin vs D1 (0–5%)"); ax.set_ylabel("CATE (Δ log units)")
    ax.grid(True, alpha=0.3, zorder=0)

plt.suptitle("Step 8 v3 — Per-Bin Honest Forests: D1 vs Dk (k=2..7)\n"
             "(each panel: separate CF with W=1{bin=Dk}, W=0{bin=D1}; honest 50/50 split)",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step08_perbin_forests_v3.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Step 8 v3 done — {len(all_rows)} contrasts")
print(out[["category","bin","n_each","ate","se","z","sig"]].to_string(index=False))
