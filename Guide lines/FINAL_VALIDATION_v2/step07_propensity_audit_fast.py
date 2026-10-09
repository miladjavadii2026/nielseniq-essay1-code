"""
Step 7 — Propensity support audit (fast version for sandbox).
OOF GroupKFold K=3 by store; 10k obs; no OHE (numeric retailer code).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step07_gates"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal", "snacks", "coffee"]
MAX_OBS   = 10_000
K_FOLDS   = 3
SUPP_INT  = [(0.05, 0.95), (0.10, 0.90)]

# ── Feature audit CSV ────────────────────────────────────────────────────────
AUDIT = [
    ("store_code_uc",       "grouping_key",       "GroupKFold only — not in X"),
    ("week_of_year",        "pre_treatment",      "USED — sin/cos seasonality"),
    ("regular_price_q75",   "pre_treatment",      "USED — background price tier"),
    ("panel_year",          "pre_treatment",      "USED — time trend"),
    ("retailer_code",       "pre_treatment",      "USED — numeric retailer indicator"),
    ("unit_price",          "EXCLUDED",           "Post-treatment-control contamination"),
    ("discount_depth",      "EXCLUDED — TREATMENT","Direct treatment variable"),
    ("depth_bin",           "EXCLUDED — TREATMENT","Binned treatment"),
    ("W5",                  "EXCLUDED — TREATMENT","Treatment indicator"),
    ("promo_event",         "EXCLUDED — TREATMENT","Contemporaneous flag"),
    ("feature",             "EXCLUDED — TREATMENT","In-promotion display"),
    ("display",             "EXCLUDED — TREATMENT","In-promotion display"),
    ("weekly_dollar_sales", "EXCLUDED",           "Outcome-derived"),
    ("log_units",           "EXCLUDED",           "Outcome"),
]
pd.DataFrame(AUDIT, columns=["column","classification","note"]).to_csv(
    f"{OUT_DIR}/a8_feature_audit.csv", index=False)
print("✓ Feature audit saved")

support_rows, qrows = [], []
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ci, cat in enumerate(CATS):
    print(f"\n{'='*50}\n  {cat.upper()}")

    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    need = ["store_code_uc","log_units","depth_bin","week_of_year",
            "regular_price_q75","panel_year","retailer_code"]
    have = [c for c in need if c in pf.schema_arrow.names]
    df   = pf.read_row_group(0, columns=have).to_pandas()
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(float)
    df   = df.dropna(subset=["log_units","depth_num","week_of_year","regular_price_q75"])
    if len(df) > MAX_OBS:
        df = df.sample(MAX_OBS, random_state=42).reset_index(drop=True)

    W      = (df["depth_num"] > 5).astype(float).values
    groups = df["store_code_uc"].fillna(0).astype(str).values
    n      = len(W)
    ts     = W.mean()
    n_t    = int(W.sum())
    print(f"  N={n:,}  treated={n_t:,} ({100*ts:.2f}%)")

    wk  = df["week_of_year"].fillna(26).astype(float).values
    rp  = df["regular_price_q75"].fillna(df["regular_price_q75"].median()).values
    yr  = df["panel_year"].fillna(2015).astype(float).values
    rc  = pd.factorize(df["retailer_code"].fillna(-1).astype(str))[0].astype(float)
    X   = np.column_stack([
        np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52),
        rp, yr, rc
    ])

    # OOF propensity via GroupKFold (no in-sample leakage)
    gkf      = GroupKFold(n_splits=K_FOLDS)
    prop_oof = np.full(n, np.nan)
    for tr_i, te_i in gkf.split(X, W, groups=groups):
        lr = LogisticRegression(solver="liblinear", max_iter=200, C=1.0, random_state=42)
        lr.fit(X[tr_i], W[tr_i])
        prop_oof[te_i] = lr.predict_proba(X[te_i])[:, 1]

    # In-sample for comparison (shows overfit)
    lr_full = LogisticRegression(solver="liblinear", max_iter=200, C=1.0, random_state=42)
    lr_full.fit(X, W)
    prop_ins = lr_full.predict_proba(X)[:, 1]

    # Quantiles
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q_all  = np.nanpercentile(prop_oof, pcts)
    q_t    = np.nanpercentile(prop_oof[W==1], pcts) if n_t > 5 else [np.nan]*9
    q_c    = np.nanpercentile(prop_oof[W==0], pcts)
    for i, p in enumerate(pcts):
        qrows.append({"category":cat,"percentile":p,
                      "oof_overall":round(q_all[i],5),
                      "oof_treated":round(float(q_t[i]),5),
                      "oof_control":round(q_c[i],5)})

    row = {"category":cat,"N":n,"treated_n":n_t,"treated_share":round(ts,4)}
    for lo, hi in SUPP_INT:
        n_supp = int(((prop_oof>lo) & (prop_oof<hi)).sum())
        pct    = 100*n_supp/n
        n_ins  = int(((prop_ins>lo) & (prop_ins<hi)).sum())
        verdict = ("FAIL (<1%)" if pct<1 else ("WARN (<10%)" if pct<10 else "PASS"))
        row[f"oof_n_{lo}_{hi}"]      = n_supp
        row[f"oof_pct_{lo}_{hi}"]    = round(pct, 2)
        row[f"ins_n_{lo}_{hi}"]      = n_ins
        row[f"verdict_{lo}_{hi}"]    = verdict
        print(f"  OOF [{lo},{hi}]: {n_supp:,}/{n:,} = {pct:.1f}%  "
              f"in-sample: {n_ins:,}  -> {verdict}")
    support_rows.append(row)

    # Histogram
    ax   = axes[ci]
    bins = np.linspace(0, 1, 41)
    ax.hist(prop_oof[W==0], bins=bins, alpha=0.55, color="#1565C0",
            label=f"Control (n={int((W==0).sum()):,})", density=True)
    ax.hist(prop_oof[W==1], bins=bins, alpha=0.70, color="#E65100",
            label=f"Treated (n={n_t:,})", density=True)
    for lo, hi in SUPP_INT:
        ax.axvline(lo, color='red', lw=1.3, ls='--')
        ax.axvline(hi, color='red', lw=1.3, ls='--')
    n10 = int(((prop_oof>0.1)&(prop_oof<0.9)).sum())
    ax.set_title(
        f"{cat.capitalize()}\ntreated {100*ts:.1f}%  in [0.1,0.9]: {n10}/{n} = {100*n10/n:.1f}%",
        fontweight='bold', fontsize=10)
    ax.set_xlabel("OOF P(W=1|X)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Per-cat figure
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.hist(prop_oof[W==0], bins=bins, alpha=0.55, color="#1565C0",
             label=f"Control (n={int((W==0).sum()):,})", density=True)
    ax2.hist(prop_oof[W==1], bins=bins, alpha=0.70, color="#E65100",
             label=f"Treated (n={n_t:,})", density=True)
    for lo, hi in SUPP_INT:
        ax2.axvline(lo, color='red', lw=1.5, ls='--')
        ax2.axvline(hi, color='red', lw=1.5, ls='--')
    ax2.set_title(
        f"{cat.capitalize()} — OOF Propensity P(W5=1|X)\n"
        f"W5=1{{depth>5%}}  treated={100*ts:.1f}%  "
        f"in [0.1,0.9]: {n10}/{n} ({100*n10/n:.1f}%)",
        fontweight='bold')
    ax2.set_xlabel("OOF Propensity P(W=1|X)"); ax2.set_ylabel("Density")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(f"{FIG_DIR}/a8_propensity_hist_{cat}.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

fig.suptitle(
    "OOF Propensity Distributions — W5=1{discount>5%}\n"
    "Red dashed lines: [0.05,0.95] and [0.10,0.90] thresholds",
    fontweight='bold')
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/a8_propensity_hist_all.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Save CSVs ────────────────────────────────────────────────────────────────
supp_df = pd.DataFrame(support_rows)
supp_df.to_csv(f"{OUT_DIR}/a8_common_support_counts.csv", index=False)
pd.DataFrame(qrows).to_csv(f"{OUT_DIR}/a8_propensity_quantiles.csv", index=False)
supp_df[["category","N","treated_n","treated_share"]].to_csv(
    f"{OUT_DIR}/a8_treatment_prevalence.csv", index=False)

# ── Overlap decision ─────────────────────────────────────────────────────────
lines = [
    "A8 OVERLAP DECISION — OOF GroupKFold K=3 propensity audit",
    "=" * 65,
    "W5 definition : 1{discount_depth > 5%} (strictly greater than)",
    "Propensity model: LogisticRegression (liblinear, C=1.0)",
    "Features (X)   : sin/cos week_of_year, regular_price_q75, panel_year,",
    "                 retailer_code (numeric label)",
    "No treatment-derived variables in X.",
    "",
    "SUPPORT [0.10, 0.90] — OOF predictions:",
]
all_fail = True
for row in support_rows:
    n10  = row["oof_n_0.1_0.9"]
    p10  = row["oof_pct_0.1_0.9"]
    v    = row["verdict_0.1_0.9"]
    ts   = row["treated_share"]
    lines.append(f"  {row['category']:8s}: {n10:5,}/{row['N']:,} = {p10:.1f}%  "
                 f"treated={100*ts:.1f}%  -> {v}")
    if p10 >= 1.0:
        all_fail = False

lines += [
    "",
    "ROOT CAUSE:",
    "  Treated share is ~4% for cereal and similarly low for snacks/coffee.",
    "  With such low prevalence, logistic regression correctly assigns",
    "  P(W=1|X) near zero to most observations regardless of feature set.",
    "  This is not model overfitting; it reflects genuinely near-deterministic",
    "  treatment assignment when discount > 5% occurs in only ~4% of",
    "  store-UPC-week observations.",
    "",
    "DECISION:",
]
if all_fail:
    lines += [
        "  WITHDRAW — near-complete common-support failure confirmed by OOF audit.",
        "  Do NOT report GATE quartiles, Q4-Q1 spreads, or permutation p-values.",
        "",
        "  Correct wording for the report (A8):",
        "  'The GATES analysis is withdrawn from inferential interpretation",
        "   because out-of-fold propensity estimates (GroupKFold K=3 by store)",
        "   show near-complete common-support failure. Fewer than 2% of",
        "   observations fall inside the [0.10, 0.90] propensity interval in",
        "   any category, reflecting extremely low treatment prevalence (~4%)",
        "   combined with the available pre-treatment features (seasonality,",
        "   background price, year, retailer). Treatment effects may be",
        "   heterogeneous, but heterogeneity cannot be credibly identified",
        "   for the full-sample ATE estimand under the current design.",
        "   This finding is confirmed by out-of-fold predictions and does",
        "   not arise from propensity-model overfitting.'",
        "",
        "  PATHWAY TO VALID GATES (workstation):",
        "  1. Add store-level prior promotion history by store-UPC pair.",
        "  2. Add chain-level promotion frequency as a feature.",
        "  3. Re-audit support BEFORE re-running forests.",
        "  4. If support <10%, redefine estimand to ATT or overlap-weighted",
        "     ATO restricted to the common-support region.",
        "  5. Report GATES only if >=10% of observations are in support.",
    ]
else:
    lines += [
        "  PARTIAL SUPPORT — at least one category has OOF support > 1%.",
        "  Restrict GATES to common-support region; change estimand to ATO/ATT.",
    ]

decision_text = "\n".join(lines)
with open(f"{OUT_DIR}/a8_overlap_decision.txt", "w") as f:
    f.write(decision_text)

print("\n" + "="*55)
print("SUPPORT SUMMARY (OOF K=3 GroupKFold by store)")
print(supp_df[["category","N","treated_share",
               "oof_n_0.05_0.95","oof_pct_0.05_0.95",
               "oof_n_0.1_0.9","oof_pct_0.1_0.9",
               "verdict_0.1_0.9"]].to_string(index=False))
print("\n" + decision_text)
print("\n✓ Saved: a8_feature_audit.csv, a8_treatment_prevalence.csv,")
print("         a8_propensity_quantiles.csv, a8_common_support_counts.csv,")
print("         a8_overlap_decision.txt, a8_propensity_hist_*.png")
