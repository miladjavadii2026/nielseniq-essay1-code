"""Step 1 v2: FE Wald contrasts + cross-category pair tests.
Fast: one-way store demeaning, 25k obs subsample per category, store-clustered SE.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings, scipy.stats as st
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step01_wald_fix"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
BIN_LABS  = {1:"D1\n0–5%",2:"D2\n5–10%",3:"D3\n10–20%",
             4:"D4\n20–30%",5:"D5\n30–40%",6:"D6\n40–50%",7:"D7\n≥50%"}
MAX_OBS   = 25000

def load_small(cat):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","depth_bin","week_of_year"]
    have = [c for c in want if c in pf.schema_arrow.names]
    df   = pf.read_row_group(0, columns=have).to_pandas()
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(float, errors='ignore')
    df = df[pd.to_numeric(df["depth_num"],errors="coerce").notna()].copy()
    df["depth_num"] = df["depth_num"].astype(int)
    df = df[df["depth_num"].between(1,7)].dropna(subset=["log_units"]).copy()
    if len(df)>MAX_OBS: df = df.sample(MAX_OBS, random_state=42)
    return df

def within_store(y, X, store_codes):
    """Demean Y and X by store. Fast O(N) approach."""
    y_ = y.copy().astype(float)
    X_ = X.copy().astype(float)
    for s in np.unique(store_codes):
        m = store_codes==s
        y_[m] -= y_[m].mean()
        X_[m] -= X_[m].mean(axis=0)
    return y_, X_

def store_cluster_se(X, resid, store_codes):
    bread  = np.linalg.pinv(X.T @ X)
    meat   = np.zeros((X.shape[1],X.shape[1]))
    stores = np.unique(store_codes)
    S = len(stores); n = len(resid); p = X.shape[1]
    for s in stores:
        m = store_codes==s; score = X[m].T @ resid[m]
        meat += np.outer(score, score)
    scale = S/(S-1) * (n-1)/(n-p)
    vcov  = scale * bread @ meat @ bread
    return np.sqrt(np.diag(vcov)), vcov

results = {}
for cat in CATS:
    print(f"\n{cat.upper()}")
    df   = load_small(cat)
    sc   = df["store_code_uc"].astype("category").cat.codes.values
    y    = df["log_units"].values
    # Depth dummies D2-D7 (D1=ref)
    Xcols = [f"D{b}" for b in range(2,8)]
    X     = np.column_stack([(df["depth_num"]==b).astype(float).values for b in range(2,8)])
    # Also add week_of_year sin/cos to control for seasonality
    wk    = df["week_of_year"].fillna(26).values
    Xw    = np.column_stack([np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52)])
    X_full = np.hstack([X, Xw])
    
    yd, Xd = within_store(y, X_full, sc)
    coefs,_,_,_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ coefs
    ses, vcov = store_cluster_se(Xd, resid, sc)
    S = len(np.unique(sc))
    
    row = {"category":cat,"n_obs":len(df),"n_stores":S}
    for i,b in enumerate(range(2,8)):
        c = float(coefs[i]); se = float(ses[i])
        t = c/se; pv = float(2*st.t.sf(abs(t),S-1))
        sig = "***" if pv<.001 else ("**" if pv<.01 else ("*" if pv<.05 else "n.s."))
        row.update({f"D{b}_coef":round(c,4),f"D{b}_se":round(se,4),
                    f"D{b}_t":round(t,3),f"D{b}_p":round(pv,5),f"D{b}_sig":sig})
        print(f"  D{b}: coef={c:.4f} SE={se:.4f} t={t:.3f} {sig}")
    
    # Monotonicity check
    clist = [row[f"D{b}_coef"] for b in range(2,8)]
    mono  = all(clist[i]<clist[i+1] for i in range(len(clist)-1))
    row["monotone_D2_D7"] = mono
    print(f"  Monotone D2→D7: {mono}")
    results[cat] = row

res_df = pd.DataFrame(results).T.reset_index(drop=True)
res_df.to_csv(f"{OUT_DIR}/step01_fe_coefs_v2.csv", index=False)

# ── Cross-category Wald tests (z-test from independent samples) ──────────────
cross_rows = []
pairs = [("cereal","snacks"),("cereal","coffee"),("snacks","coffee")]
for b in [6,7]:
    for c1,c2 in pairs:
        r1,r2 = results[c1],results[c2]
        d = r1[f"D{b}_coef"] - r2[f"D{b}_coef"]
        se= np.sqrt(r1[f"D{b}_se"]**2 + r2[f"D{b}_se"]**2)
        z = d/se; pv = float(2*(1-st.norm.cdf(abs(z))))
        sig= "***" if pv<.001 else ("**" if pv<.01 else ("*" if pv<.05 else "n.s."))
        cross_rows.append({"bin":f"D{b}","cat1":c1,"cat2":c2,
                           "coef_1":r1[f"D{b}_coef"],"coef_2":r2[f"D{b}_coef"],
                           "diff":round(d,4),"se":round(se,4),"z":round(z,3),
                           "p":round(pv,5),"sig":sig})
        print(f"D{b} {c1}−{c2}: diff={d:.4f} z={z:.3f} {sig}")

cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(f"{OUT_DIR}/step01_cross_category_wald.csv", index=False)

# ── Figure 1: Per-bin coefs ───────────────────────────────────────────────────
fig,axes = plt.subplots(1,3,figsize=(15,5),sharey=False)
CAT_C = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for ax,cat in zip(axes,CATS):
    row  = results[cat]
    xs   = list(range(1,8))
    ys   = [0.0]+[row[f"D{b}_coef"] for b in range(2,8)]
    ses_ = [0.0]+[row[f"D{b}_se"]   for b in range(2,8)]
    sigs_= [""]+[row[f"D{b}_sig"]   for b in range(2,8)]
    c    = CAT_C[cat]
    ax.bar(xs, ys, color=c, alpha=0.65, width=0.55)
    ax.errorbar(xs[1:], ys[1:], yerr=[1.96*s for s in ses_[1:]],
                fmt='o', color='k', ms=7, capsize=6, lw=1.5, zorder=5)
    for x,y_,sig in zip(xs,ys,sigs_):
        if sig: ax.text(x,y_+0.03,sig,ha='center',fontsize=9,fontweight='bold')
    mono = row.get("monotone_D2_D7","?")
    ax.set_title(f"{cat.capitalize()} (N={row['n_obs']:,}, S={row['n_stores']})\n"
                 f"Monotone D2→D7: {mono}",fontsize=10,fontweight='bold')
    ax.set_xticks(xs); ax.set_xticklabels([BIN_LABS[b] for b in xs],fontsize=7.5)
    ax.set_xlabel("Depth bin"); ax.set_ylabel("β_Dk vs D1 (log units)")
    ax.axhline(0,color='k',lw=0.8); ax.grid(True,alpha=0.3)

plt.suptitle("Step 1 v2 — FE Wald Contrasts D1–D7 (D1=ref)\n"
             "One-way store FE + seasonality; store-clustered SE; 25k obs subsample",
             fontsize=11,fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step01_fe_coefs_v2.png",dpi=150,bbox_inches='tight')
plt.close()

# ── Figure 2: Cross-category contrasts ───────────────────────────────────────
fig,axes = plt.subplots(1,2,figsize=(11,5))
PAIR_C = [("#7C3AED","cereal vs snacks"),("#0891B2","cereal vs coffee"),("#D97706","snacks vs coffee")]
for ax,b in zip(axes,[6,7]):
    sub = cross_df[cross_df["bin"]==f"D{b}"]
    ys  = sub["diff"].values; ses_c = sub["se"].values
    labs= [f"{r.cat1.capitalize()} − {r.cat2.capitalize()}" for _,r in sub.iterrows()]
    cols= [p[0] for p in PAIR_C]
    ax.barh(range(len(ys)), ys, color=cols, alpha=0.7, height=0.5)
    ax.errorbar(ys, range(len(ys)), xerr=1.96*ses_c,
                fmt='o', color='k', ms=8, capsize=7, lw=1.5)
    for i,(d,sig) in enumerate(zip(ys, sub["sig"])):
        col = "darkgreen" if sig!="n.s." else "red"
        ax.text(d+0.005*(1 if d>=0 else -1), i, sig, va='center',
                fontsize=12, fontweight='bold', color=col)
    ax.axvline(0,color='k',lw=1.5,ls='--')
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(labs,fontsize=10)
    ax.set_title(f"Cross-category Wald at D{b}\nH₀: β_cat1 = β_cat2",
                 fontsize=10,fontweight='bold')
    ax.set_xlabel("Δ coefficient (log units)"); ax.grid(True,alpha=0.3)

plt.suptitle("Step 1 v2 — Cross-Category Pairwise Wald Tests at D6 & D7",
             fontsize=11,fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step01_cross_category_wald.png",dpi=150,bbox_inches='tight')
plt.close()
print("\n✓ All Step 1 v2 outputs saved")
