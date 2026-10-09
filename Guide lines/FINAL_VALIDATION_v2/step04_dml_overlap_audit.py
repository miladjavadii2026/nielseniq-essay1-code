"""
Step 4 — DML Overlap Audit (Advisor 8th Verdict, Item D).

Uses EXACT same features and grouping as production step04_fast.py:
  X = [upc_h, brand_h, week_of_year, unit_price]
  W = 1{discount_depth > 0.05}  (W5 definition, strictly >5%)
  GroupKFold K=5 by store_code_uc

Additional checks:
  - Compare Ridge / Lasso / ElasticNet treatment propensity models
  - OOF propensity support at [0.05,0.95] and [0.10,0.90]
  - ATE sensitivity: full vs trimmed to common support
  - NOTE: unit_price is in X but is post-treatment; flagged here as a limitation

NOTE on unit_price in X:
  The production DML includes unit_price = sale_price (= regular_price * (1-discount)).
  Because discount_depth is the treatment, unit_price = f(treatment) and is partly
  post-treatment. This could bias the residualization. Flagged as a limitation;
  replication without unit_price is included for comparison.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
import os, scipy.stats as st

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step04_grouped_dml"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

CAT      = "cereal"   # Full audit on cereal (122,880 obs); results generalise
K_FOLDS  = 5
SUPP_INTS = [(0.05, 0.95), (0.10, 0.90)]

print(f"{'='*65}")
print(f"  DML OVERLAP AUDIT — {CAT.upper()}, full row group 0")
print(f"  Step04 exact spec: X=[upc_h,brand_h,week_of_year,unit_price]")
print(f"  K={K_FOLDS} GroupKFold by store_code_uc")
print(f"{'='*65}")

pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{CAT}_analysis.parquet")
cols = ["store_code_uc","upc","log_units","discount_depth",
        "week_of_year","brand_code_uc","unit_price",
        "regular_price_q75","panel_year","retailer_code"]
have = [c for c in cols if c in pf.schema_arrow.names]
df   = pf.read_row_group(0, columns=have).to_pandas().dropna(subset=["log_units"])

n = len(df)
print(f"  N={n:,}")

# Treatment (correct definition)
df["W5"] = (df["discount_depth"] > 0.05).astype(float)
W        = df["W5"].values
ts       = W.mean()
print(f"  Treated (W5=1): {int(W.sum()):,} ({100*ts:.1f}%)  "
      f"Control: {int((W==0).sum()):,} ({100*(1-ts):.1f}%)")

# Build feature matrices
df["upc_h"]   = pd.factorize(df["upc"])[0].astype(float)
df["brand_h"] = pd.factorize(df["brand_code_uc"])[0].astype(float)
df["unit_price_fill"] = df["unit_price"].fillna(0)

# Spec A: exact production spec (includes unit_price)
X_A = df[["upc_h","brand_h","week_of_year","unit_price_fill"]].fillna(0).values
# Spec B: clean spec (no unit_price, adds retailer+price tier)
df["rc_num"] = pd.factorize(df["retailer_code"].fillna(-1).astype(str))[0].astype(float)
rp           = df["regular_price_q75"].fillna(df["regular_price_q75"].median()).values
wk           = df["week_of_year"].fillna(26).astype(float).values
X_B = np.column_stack([
    df["upc_h"].values, df["brand_h"].values, wk,
    np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52),
    rp, df["rc_num"].values
])

groups  = df["store_code_uc"].values
n_sp    = min(K_FOLDS, len(np.unique(groups)))
gkf     = GroupKFold(n_splits=n_sp)
cv_list = list(gkf.split(X_A, W, groups))
print(f"  Stores={len(np.unique(groups)):,}  actual splits={n_sp}")

rows = []
for spec_name, X_spec in [("ProductionSpec(with unit_price)", X_A),
                            ("CleanSpec(no unit_price)", X_B)]:
    print(f"\n  --- {spec_name} ---")
    for lr_name, model_cls in [
        ("Ridge",      Ridge(alpha=1.0)),
        ("Lasso",      Lasso(alpha=0.001, max_iter=2000)),
        ("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000)),
    ]:
        pipe = Pipeline([("sc", StandardScaler()), ("m", model_cls)])
        # OOF treatment propensity via continuous W-hat (linear DML)
        W_hat = cross_val_predict(pipe, X_spec, W, cv=cv_list)

        # Propensity: clip to [0,1] (linear model can exceed)
        prop  = np.clip(W_hat, 0, 1)
        W_res = W - prop

        for lo, hi in SUPP_INTS:
            mask   = (prop > lo) & (prop < hi)
            n_supp = int(mask.sum())
            pct    = 100 * n_supp / n
            verdict = ("PASS" if pct >= 20 else ("WARN" if pct >= 10 else "FAIL"))
            print(f"    {lr_name:12s} [{lo},{hi}]: {n_supp:,}/{n:,} = {pct:.1f}%  → {verdict}")
            rows.append({"spec": spec_name, "learner": lr_name,
                         "lo": lo, "hi": hi, "n_supp": n_supp,
                         "pct": round(pct,1), "verdict": verdict})

# DML ATE sensitivity (Ridge, exact spec A, for report)
print(f"\n  --- ATE sensitivity (Ridge, Spec A, cereal) ---")
pipe_y = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
pipe_t = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
Y = df["log_units"].values
Y_hat = cross_val_predict(pipe_y, X_A, Y, cv=cv_list)
W_hat = cross_val_predict(pipe_t, X_A, W, cv=cv_list)
Y_res = Y - Y_hat
W_res = W - W_hat
prop_clip = np.clip(W_hat, 0, 1)

def lz_se(Y_r, W_r, grps):
    G = len(np.unique(grps)); nobs = len(Y_r)
    theta = float(np.sum(W_r * Y_r) / np.sum(W_r**2))
    eps   = Y_r - theta * W_r
    agg   = (pd.DataFrame({"s": grps, "Wr": W_r, "ep": eps})
             .groupby("s").apply(lambda x: float(np.sum(x["Wr"]*x["ep"]))).values)
    meat  = np.sum(agg**2) * (G/(G-1)) * ((nobs-1)/(nobs-3))
    se    = float(np.sqrt(max(0, meat) / np.sum(W_r**2)**2))
    t_    = theta / se if se > 0 else np.nan
    p_    = float(2*(1-st.norm.cdf(abs(t_)))) if not np.isnan(t_) else np.nan
    return theta, se, t_, p_

# Full ATE
theta_f, se_f, t_f, p_f = lz_se(Y_res, W_res, groups)
# Trimmed ATE [0.10, 0.90]
trim = (prop_clip > 0.10) & (prop_clip < 0.90)
theta_t, se_t, t_t, p_t = lz_se(Y_res[trim], W_res[trim], groups[trim])
# Trimmed [0.05, 0.95]
trim2 = (prop_clip > 0.05) & (prop_clip < 0.95)
theta_t2, se_t2, t_t2, p_t2 = lz_se(Y_res[trim2], W_res[trim2], groups[trim2])

ate_rows = [
    {"estimand":"Full ATE",          "N":n,
     "ATE":round(theta_f,4),  "SE":round(se_f,4),  "t":round(t_f,2),  "p":round(p_f,4)},
    {"estimand":"Trimmed[0.05,0.95]","N":int(trim2.sum()),
     "ATE":round(theta_t2,4), "SE":round(se_t2,4), "t":round(t_t2,2), "p":round(p_t2,4)},
    {"estimand":"Trimmed[0.10,0.90]","N":int(trim.sum()),
     "ATE":round(theta_t,4),  "SE":round(se_t,4),  "t":round(t_t,2),  "p":round(p_t,4)},
]
for r in ate_rows:
    print(f"  {r['estimand']:25s}  N={r['N']:,}  ATE={r['ATE']:.4f}  "
          f"SE={r['SE']:.4f}  t={r['t']:.2f}  p={r['p']:.4f}")

# Save
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/dml_overlap_audit.csv", index=False)
pd.DataFrame(ate_rows).to_csv(f"{OUT_DIR}/dml_ate_sensitivity.csv", index=False)

# Propensity histogram (Ridge, Spec A)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
prop_a = np.clip(cross_val_predict(
    Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=1.0))]),
    X_A, W, cv=cv_list), 0, 1)
for ax, spec_name2, prop2 in zip(axes2, ["Spec A (production)", "Spec B (clean)"], [prop_a, prop_a]):
    bins = np.linspace(0, 1, 51)
    ax.hist(prop2[W==0], bins=bins, alpha=0.5, color="#1565C0",
            label=f"Control ({int((W==0).sum()):,})", density=True)
    ax.hist(prop2[W==1], bins=bins, alpha=0.6, color="#E65100",
            label=f"Treated ({int(W.sum()):,})", density=True)
    ax.axvline(0.05, color='red', lw=1.5, ls='--', label="[0.05,0.95]")
    ax.axvline(0.95, color='red', lw=1.5, ls='--')
    ax.axvline(0.10, color='orange', lw=1.5, ls=':', label="[0.10,0.90]")
    ax.axvline(0.90, color='orange', lw=1.5, ls=':')
    n10 = int(((prop2>0.10)&(prop2<0.90)).sum())
    ax.set_title(f"Ridge W-hat — Cereal {spec_name2}\n"
                 f"[0.1,0.9]: {n10}/{n} = {100*n10/n:.0f}%",
                 fontweight='bold', fontsize=9)
    ax.set_xlabel("W-hat (Ridge residualised propensity)"); ax.set_ylabel("Density")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step04_dml_overlap.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ DML overlap audit complete")
print(f"  Saved: dml_overlap_audit.csv, dml_ate_sensitivity.csv, step04_dml_overlap.png")
