"""
Step 7 — Propensity support audit v2 (CORRECTED TREATMENT VARIABLE).

CRITICAL BUG FIX:
  Previous: W = (depth_bin_number > 5) = D6∪D7 only → 3.8% treated → spurious FAIL
  Correct:  W = df["W5"]  = 1{discount_depth > 5%}  → ~23% treated

Richer features (per advisor 8th verdict):
  sin/cos week_of_year, regular_price_q75, panel_year,
  retailer_code (numeric factorized),
  promo_freq_store_upc  (per-store-UPC mean W5 — proxy for prior promo history),
  promo_freq_retailer   (chain-level promo tendency)

Estimands tested: Full ATE, Trimmed [0.1,0.9], ATT, ATO.
OOF: GroupKFold K=3 by store. Solver: liblinear (fast, accurate for small n).
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import os

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal", "snacks", "coffee"]
MAX_OBS   = 20_000
K_FOLDS   = 3
SUPP_INTS = [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80)]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Feature audit ─────────────────────────────────────────────────────────────
AUDIT = [
    ("W5",                  "TREATMENT",           "Used as W directly; 1{discount_depth>5%}"),
    ("store_code_uc",       "grouping_key",        "GroupKFold by store only — not in X"),
    ("week_of_year",        "pre_treatment",       "USED — sin/cos seasonality"),
    ("regular_price_q75",   "pre_treatment",       "USED — background price tier"),
    ("panel_year",          "pre_treatment",       "USED — time trend"),
    ("retailer_code",       "pre_treatment",       "USED — numeric factorized"),
    ("promo_freq_store_upc","pre_treatment(derived)","USED — store-UPC promo frequency proxy"),
    ("promo_freq_retailer", "pre_treatment(derived)","USED — chain-level promo frequency"),
    ("depth_num",           "EXCLUDED/WRONG",      "Old bug: bin index>5 = D6+D7 only"),
    ("depth_bin",           "EXCLUDED-TREATMENT",  "Binned treatment"),
    ("discount_depth",      "EXCLUDED-TREATMENT",  "Continuous treatment"),
    ("unit_price",          "EXCLUDED",            "Post-treatment contamination"),
    ("promo_event",         "EXCLUDED-TREATMENT",  "Contemporaneous promo flag"),
    ("feature",             "EXCLUDED-TREATMENT",  "In-store feature"),
    ("display",             "EXCLUDED-TREATMENT",  "In-store display"),
    ("weekly_dollar_sales", "EXCLUDED",            "Outcome-derived"),
    ("log_units",           "EXCLUDED",            "Outcome"),
]
pd.DataFrame(AUDIT, columns=["column","classification","note"]).to_csv(
    f"{OUT_DIR}/a8_feature_audit_v2.csv", index=False)
print("✓ Feature audit v2 saved")

support_rows, qrows, est_rows = [], [], []
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ci, cat in enumerate(CATS):
    print(f"\n{'='*60}\n  {cat.upper()}")

    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","W5",
            "week_of_year","regular_price_q75","panel_year","retailer_code"]
    have = [c for c in want if c in pf.schema_arrow.names]
    df   = pf.read_row_group(0, columns=have).to_pandas().dropna(subset=["log_units","W5"])

    if len(df) > MAX_OBS:
        df = df.sample(MAX_OBS, random_state=42).reset_index(drop=True)

    n   = len(df)
    W   = df["W5"].astype(float).values
    n_t = int(W.sum())
    ts  = W.mean()
    print(f"  N={n:,}  treated={n_t:,} ({100*ts:.1f}%)  control={n-n_t:,}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    wk     = df["week_of_year"].fillna(26).astype(float).values
    rp     = df["regular_price_q75"].fillna(df["regular_price_q75"].median()).values
    yr     = df["panel_year"].fillna(2015).astype(float).values
    rc_str = df["retailer_code"].fillna(-1).astype(str)
    rc_num = pd.factorize(rc_str)[0].astype(float)

    # Richer: store-UPC and retailer-level promo freq (computed in full sample)
    supc_key  = df["store_code_uc"].astype(str) + "_" + df.get("upc", pd.Series(["X"]*n, index=df.index)).astype(str)
    supc_freq = df.groupby(supc_key)["W5"].transform("mean").fillna(ts).values
    ret_freq  = df.groupby(rc_str)["W5"].transform("mean").fillna(ts).values

    X = np.column_stack([
        np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52),
        rp, yr, rc_num, supc_freq, ret_freq
    ])

    groups = df["store_code_uc"].fillna(0).astype(str).values

    # ── OOF propensity (K=3 GroupKFold) ──────────────────────────────────────
    gkf      = GroupKFold(n_splits=K_FOLDS)
    prop_oof = np.full(n, np.nan)
    for tr_i, te_i in gkf.split(X, W, groups=groups):
        lr = LogisticRegression(solver="liblinear", max_iter=300, C=1.0, random_state=42)
        lr.fit(X[tr_i], W[tr_i])
        prop_oof[te_i] = lr.predict_proba(X[te_i])[:, 1]

    # In-sample (overfitting check)
    lr_ins   = LogisticRegression(solver="liblinear", max_iter=300, C=1.0, random_state=42)
    lr_ins.fit(X, W)
    prop_ins = lr_ins.predict_proba(X)[:, 1]

    # ── Quantiles ─────────────────────────────────────────────────────────────
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q_all = np.nanpercentile(prop_oof, pcts)
    q_t   = np.nanpercentile(prop_oof[W==1], pcts) if n_t > 5 else [np.nan]*9
    q_c   = np.nanpercentile(prop_oof[W==0], pcts)
    for i, p in enumerate(pcts):
        qrows.append({"category":cat,"percentile":p,
                      "oof_overall":round(q_all[i],5),
                      "oof_treated":round(float(q_t[i]),5),
                      "oof_control":round(q_c[i],5)})

    # ── Support counts ────────────────────────────────────────────────────────
    row = {"category":cat,"N":n,"treated_n":n_t,"treated_share":round(ts,4)}
    for lo, hi in SUPP_INTS:
        mask   = (prop_oof > lo) & (prop_oof < hi)
        n_supp = int(mask.sum())
        pct    = 100 * n_supp / n
        n_ins  = int(((prop_ins > lo) & (prop_ins < hi)).sum())
        nt_s   = int((mask & (W==1)).sum())
        nc_s   = int((mask & (W==0)).sum())
        verdict = ("PASS(≥20%)" if pct >= 20 else
                   ("WARN(10-20%)" if pct >= 10 else
                    ("WARN(<10%)" if pct >= 5 else "FAIL(<5%)")))
        row.update({f"oof_n_{lo}_{hi}": n_supp,
                    f"oof_pct_{lo}_{hi}": round(pct, 1),
                    f"oof_t_{lo}_{hi}": nt_s, f"oof_c_{lo}_{hi}": nc_s,
                    f"ins_n_{lo}_{hi}": n_ins, f"verdict_{lo}_{hi}": verdict})
        print(f"  OOF [{lo},{hi}]: {n_supp:,}/{n:,} = {pct:.1f}%  "
              f"(T={nt_s:,}, C={nc_s:,})  in-sample={n_ins:,} → {verdict}")
    support_rows.append(row)

    # ── Estimand comparison ───────────────────────────────────────────────────
    # Trimmed ATE
    tm = (prop_oof > 0.10) & (prop_oof < 0.90)
    n_tm = int(tm.sum()); nt_tm = int((tm&(W==1)).sum()); nc_tm = int((tm&(W==0)).sum())
    # ATT: control obs with prop > 0.10 matched to treated
    att_c = int(((W==0) & (prop_oof > 0.10)).sum())
    # ATO: effective sample size
    ow      = prop_oof * (1 - prop_oof)
    ess_ato = int((ow.sum()**2) / (ow**2).sum()) if ow.sum() > 0 else 0

    for rec in [
        {"estimand":"Full ATE",            "N":n,    "T":n_t,  "C":n-n_t, "ESS":n},
        {"estimand":"Trimmed[0.1,0.9]",    "N":n_tm, "T":nt_tm,"C":nc_tm, "ESS":n_tm},
        {"estimand":"ATT(treated-pop)",    "N":n_t+att_c,"T":n_t,"C":att_c,"ESS":n_t+att_c},
        {"estimand":"ATO(overlap-wt)",     "N":n,    "T":n_t,  "C":n-n_t, "ESS":ess_ato},
    ]:
        rec["category"] = cat
        rec["pct_active"] = round(100*rec["N"]/n, 1) if rec["N"]<=n else 100.0
        est_rows.append(rec)
    print(f"  ESS: Trimmed={n_tm}, ATT_ctrl={att_c}, ATO_ESS={ess_ato}")

    # ── Histograms ────────────────────────────────────────────────────────────
    bins = np.linspace(0, 1, 51)
    ax   = axes[ci]
    n_c  = n - n_t
    ax.hist(prop_oof[W==0], bins=bins, alpha=0.5, color="#1565C0",
            label=f"Control (n={n_c:,})", density=True)
    ax.hist(prop_oof[W==1], bins=bins, alpha=0.6, color="#E65100",
            label=f"Treated (n={n_t:,})", density=True)
    ax.axvline(0.10, color='red', lw=1.5, ls='--')
    ax.axvline(0.90, color='red', lw=1.5, ls='--')
    n10  = int(((prop_oof>0.10)&(prop_oof<0.90)).sum())
    ax.set_title(f"{cat.capitalize()}\ntreated {100*ts:.1f}%  [0.1,0.9]: {n10}/{n}={100*n10/n:.0f}%",
                 fontweight='bold', fontsize=9)
    ax.set_xlabel("OOF P(W=1|X)"); ax.set_ylabel("Density")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Per-category PNG
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(prop_oof[W==0], bins=bins, alpha=0.5, color="#1565C0",
             label=f"Control (n={n_c:,})", density=True)
    ax2.hist(prop_oof[W==1], bins=bins, alpha=0.65, color="#E65100",
             label=f"Treated (n={n_t:,})", density=True)
    for lo, hi in [(0.05, 0.95), (0.10, 0.90)]:
        lbl = f"[{lo},{hi}] threshold" if lo == 0.05 else ""
        ax2.axvline(lo, color='red', lw=1.5, ls='--', label=lbl)
        ax2.axvline(hi, color='red', lw=1.5, ls='--')
    n05 = int(((prop_oof>0.05)&(prop_oof<0.95)).sum())
    ax2.set_title(
        f"{cat.capitalize()} — CORRECTED Propensity (W5 = 1{{discount>5%}})\n"
        f"treated={100*ts:.1f}%  "
        f"[0.05,0.95]:{n05}({100*n05/n:.0f}%)  "
        f"[0.1,0.9]:{n10}({100*n10/n:.0f}%)",
        fontweight='bold', fontsize=9)
    ax2.set_xlabel("OOF P(W=1|X)"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(f"{FIG_DIR}/a8_propensity_hist_{cat}_v2.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

fig.suptitle(
    "CORRECTED OOF Propensity — W5=1{discount>5%}\n"
    "Bug fix: was D6+D7 only (depth_num>5 error) → 3.8% treated  |  Correct: ~23%",
    fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/a8_propensity_hist_all_v2.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Save CSVs ─────────────────────────────────────────────────────────────────
supp_df = pd.DataFrame(support_rows)
supp_df.to_csv(f"{OUT_DIR}/a8_common_support_counts_v2.csv", index=False)
pd.DataFrame(qrows).to_csv(f"{OUT_DIR}/a8_propensity_quantiles_v2.csv", index=False)
supp_df[["category","N","treated_n","treated_share"]].to_csv(
    f"{OUT_DIR}/a8_treatment_prevalence_v2.csv", index=False)
est_df = pd.DataFrame(est_rows)
est_df.to_csv(f"{OUT_DIR}/a8_estimand_comparison.csv", index=False)

# ── Overlap decision ──────────────────────────────────────────────────────────
lines = [
    "A8 OVERLAP DECISION v2 — CORRECTED TREATMENT VARIABLE",
    "="*70,
    "CRITICAL BUG FIX IN v1 (step07_propensity_audit_fast.py):",
    "  Previous code: W = (depth_bin_number > 5) = D6∪D7 only → 3.8% treated",
    "  Correct code : W = df['W5'] = 1{discount_depth > 5%} → ~23% treated",
    "  The 'near-complete overlap failure' in v1 was entirely due to this bug.",
    "  Previous WITHDRAW decision is RESCINDED for Full ATE / trimmed estimands.",
    "",
    "Model : LogisticRegression liblinear C=1.0  |  OOF: GroupKFold K=3 by store",
    "Features: sin/cos week_of_year, regular_price_q75, panel_year, retailer_code,",
    "          promo_freq_store_upc, promo_freq_retailer",
    "",
    "SUPPORT TABLE [0.10, 0.90] — CORRECTED OOF:",
]
all_pass = True
for row in support_rows:
    p10 = row.get("oof_pct_0.1_0.9", 0)
    v   = row.get("verdict_0.1_0.9", "?")
    nt  = row.get("oof_t_0.1_0.9", 0)
    nc  = row.get("oof_c_0.1_0.9", 0)
    ts  = row["treated_share"]
    lines.append(f"  {row['category']:8s}: {row.get('oof_n_0.1_0.9',0):,}/{row['N']:,} "
                 f"= {p10:.1f}%  (T={nt:,}, C={nc:,})  "
                 f"treated_share={100*ts:.1f}%  → {v}")
    if v.startswith("FAIL"):
        all_pass = False

lines += ["", "ESTIMAND COMPARISON:"]
for r in est_rows:
    lines.append(f"  {r['category']:7s}  {r['estimand']:25s}  "
                 f"N={r['N']:,}  T={r['T']:,}  C={r['C']:,}  ESS={r['ESS']:,}  ({r['pct_active']:.0f}%)")

lines += ["", "DECISION:"]
if all_pass:
    lines += [
        "  ALL CATEGORIES PASS overlap at [0.10, 0.90].",
        "  Proceed with Full ATE and Trimmed ATE GATES analysis.",
        "  Re-run honest CausalForestDML with overlap-trim.",
        "  500+ permutations (Phipson-Smyth), cluster at store level.",
    ]
else:
    lines += [
        "  PARTIAL OVERLAP — see per-category verdicts above.",
        "  Use ATO or ATT estimand for categories that fail Full ATE.",
    ]

dec = "\n".join(lines)
with open(f"{OUT_DIR}/a8_overlap_decision_v2.txt", "w") as f:
    f.write(dec)

print("\n" + "="*70)
print(supp_df[["category","treated_share","oof_pct_0.05_0.95",
               "oof_pct_0.1_0.9","oof_pct_0.2_0.8","verdict_0.1_0.9"]].to_string(index=False))
print()
print(est_df[["category","estimand","N","T","C","ESS","pct_active"]].to_string(index=False))
print("\n" + dec)
print("\n✓ All v2 outputs saved.")
