"""Step 8 v2: CausalForestDML with continuous W=depth_num.
Features: week_of_year (sin/cos), regular_price_q75 (price tier proxy, pre-treatment).
Subsample 40k obs per category for speed. All 7 bins if obs>=20.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from econml.dml import CausalForestDML

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2"
FIG_DIR   = f"{OUT_DIR}/figures"
CATS      = {"cereal":1344,"snacks":1323,"coffee":1463}
D_LABELS  = {1:"D1\n0–5%",2:"D2\n5–10%",3:"D3\n10–20%",
             4:"D4\n20–30%",5:"D5\n30–40%",6:"D6\n40–50%",7:"D7\n≥50%"}
MIN_BIN   = 20
MAX_OBS   = 40000

def load_rms(cat, n_rg=2):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["log_units","depth_bin","week_of_year","regular_price_q75","brand_code_uc"]
    have = [c for c in want if c in pf.schema_arrow.names]
    frames = [pf.read_row_group(i, columns=have).to_pandas()
              for i in range(min(n_rg, pf.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["depth_num"] = df["depth_bin"].str.replace("D","",regex=False).astype(float)
    return df

all_summary = []
fig_cats    = {}

for cat in CATS:
    print(f"\n{'='*50}\n  {cat.upper()}")
    df = load_rms(cat)
    df = df.dropna(subset=["log_units","depth_num"]).copy()
    bin_cts = df["depth_num"].value_counts().sort_index()
    print("  Bin counts (full):\n", bin_cts.to_string())
    valid_bins = bin_cts[bin_cts >= MIN_BIN].index
    df = df[df["depth_num"].isin(valid_bins)].copy()
    # Subsample for speed
    if len(df) > MAX_OBS:
        df = df.sample(MAX_OBS, random_state=42).copy()
    n_v = df["depth_num"].nunique()
    print(f"  Using {len(df):,} obs | {n_v} valid bins")
    if len(df) < 200 or n_v < 2: print("  *** Insufficient"); continue

    Y = df["log_units"].values
    W = df["depth_num"].values

    # Features: sin/cos seasonality + price tier proxy (pre-treatment stable)
    wk = df["week_of_year"].fillna(26).values.astype(float)
    X_cols = [
        np.sin(2*np.pi*wk/52),
        np.cos(2*np.pi*wk/52),
    ]
    if "regular_price_q75" in df.columns:
        X_cols.append(df["regular_price_q75"].fillna(df["regular_price_q75"].median()).values)
    if "brand_code_uc" in df.columns:
        brand = pd.Categorical(df["brand_code_uc"]).codes.astype(float)
        X_cols.append(brand)
    X = np.column_stack(X_cols)

    rng = np.random.default_rng(42)
    n   = len(Y); idx = np.arange(n); rng.shuffle(idx)
    ia, ib = idx[:n//2], idx[n//2:]
    Ya,Wa,Xa = Y[ia],W[ia],X[ia]
    Yb,Wb,Xb = Y[ib],W[ib],X[ib]

    cf = CausalForestDML(
        model_y=Ridge(alpha=1.0), model_t=Ridge(alpha=1.0),
        n_estimators=100, min_samples_leaf=10,
        random_state=42, honest=True, inference=True
    )
    cf.fit(Ya, Wa, X=Xa)
    cate_b = cf.effect(Xb).flatten()
    print(f"  CATE held-out: mean={cate_b.mean():.3f} IQR=[{np.percentile(cate_b,25):.3f},{np.percentile(cate_b,75):.3f}]")

    dose_rows = []
    bins_all  = sorted(df["depth_num"].unique().tolist())
    ref_b     = min(bins_all)
    for b in bins_all:
        obs_b = int((W==b).sum())
        if b == ref_b:
            dose_rows.append({"bin":b,"label":D_LABELS.get(int(b),"D?"),
                               "ate":0.0,"se":np.nan,"ci_lo":0.0,"ci_hi":0.0,
                               "obs":obs_b,"note":"ref"}); continue
        try:
            inf_   = cf.effect_inference(Xb, T0=np.full(len(Xb),ref_b),
                                              T1=np.full(len(Xb),b))
            ate_pt = float(inf_.pred.mean())
            se_pt  = float(np.std(inf_.pred) / np.sqrt(len(inf_.pred)))
            dose_rows.append({"bin":b,"label":D_LABELS.get(int(b),"D?"),
                               "ate":round(ate_pt,4),"se":round(se_pt,4),
                               "ci_lo":round(ate_pt-1.96*se_pt,4),
                               "ci_hi":round(ate_pt+1.96*se_pt,4),
                               "obs":obs_b,"note":"ok"})
        except Exception as e:
            dose_rows.append({"bin":b,"label":D_LABELS.get(int(b),"D?"),
                               "ate":np.nan,"se":np.nan,"ci_lo":np.nan,"ci_hi":np.nan,
                               "obs":obs_b,"note":f"err:{str(e)[:30]}"})

    dr = pd.DataFrame(dose_rows)
    dr["category"] = cat
    all_summary.append(dr); fig_cats[cat] = dr
    print(dr[["bin","obs","ate","se","note"]].to_string(index=False))

# ── Figures ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(15,5),sharey=False)
cat_colors = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}

for ax, cat in zip(axes, CATS):
    dr = fig_cats.get(cat)
    if dr is None: ax.set_title(cat.capitalize()+"\n(no data)"); continue
    ok = dr[dr["note"]=="ok"]; ref=dr[dr["note"]=="ref"]
    ax.set_title(f"{cat.capitalize()}\nCausalForestDML (multi-valued W)",fontsize=10,fontweight='bold')
    for _,row in pd.concat([ref,ok]).iterrows():
        ax.bar(row["bin"], row["ate"], color=cat_colors[cat], alpha=0.65, width=0.55)
    if len(ok):
        ax.errorbar(ok["bin"].values, ok["ate"].values, yerr=1.96*ok["se"].values,
                    fmt='o',color='k',ms=7,capsize=6,lw=1.5,zorder=5)
    ax.axhline(0,color='k',lw=0.8)
    ax.set_xlabel("Depth bin"); ax.set_ylabel("CATE vs. D1 (Δ log units)")
    ax.set_xticks(sorted(dr["bin"].values))
    ax.set_xticklabels([D_LABELS.get(int(b),"D?") for b in sorted(dr["bin"].values)],fontsize=7.5)
    ax.grid(True,alpha=0.3)

plt.suptitle("Step 8 v2 — Multi-Valued CausalForestDML Dose-Response\n"
             "(honest 50/50; W=depth_num continuous; 40k subsample)",
             fontsize=11,fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step08_causal_forest_dose_response_v2.png",dpi=150,bbox_inches='tight')
plt.close()

if all_summary:
    out=pd.concat(all_summary,ignore_index=True)
    out.to_csv(f"{OUT_DIR}/step08_causal_forest_dose_response/dose_response_v2.csv",index=False)
    print("\n✓ Figure + CSV saved.")
    print(out[["category","bin","obs","ate","se","note"]].to_string(index=False))
