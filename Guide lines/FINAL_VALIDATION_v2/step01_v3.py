"""Step 1 v3: Cross-category Wald contrasts + Cochran-Q joint equality test.
Fixes:
- D6 Snacks vs Coffee z=-2.01 correctly labeled * (p≈0.044), NOT n.s.
- Joint Wald chi-squared (Cochran's Q) for H0: β_D6 equal across cats
- p-values computed and reported for ALL pair contrasts
- NOTE: separate per-category regressions (not pooled); one-way store FE;
  25k obs subsample. This is a DIAGNOSTIC, not the full three-way-FE pooled model.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step01_wald_fix"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
MAX_OBS   = 25000

def load_rms(cat):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","log_units","depth_bin","week_of_year"]
    have = [c for c in want if c in pf.schema_arrow.names]
    df   = pf.read_row_group(0, columns=have).to_pandas()
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(int)
    return df

def within_store(y, X, sc):
    y_ = y.copy().astype(float); X_ = X.copy().astype(float)
    for s in np.unique(sc):
        m=sc==s
        if m.sum()<2: continue
        y_[m]-=y_[m].mean(); X_[m]-=X_[m].mean(axis=0)
    return y_, X_

def cluster_se(X, resid, sc):
    bread=np.linalg.pinv(X.T@X); meat=np.zeros((X.shape[1],X.shape[1]))
    stores=np.unique(sc); S=len(stores); n=len(resid); p=X.shape[1]
    for s in stores:
        m=sc==s; sc_=X[m].T@resid[m]; meat+=np.outer(sc_,sc_)
    return np.sqrt(np.diag(S/(S-1)*(n-1)/(n-p)*bread@meat@bread))

def sig_label(z):
    p = 2*(1-stats.norm.cdf(abs(z)))
    s = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "n.s."))
    return p, s

# ── Estimate per-category ────────────────────────────────────────────────────
results = {}   # cat → {bin: (coef, se)}
for cat in CATS:
    df = load_rms(cat).dropna(subset=["log_units","depth_num"])
    if len(df)>MAX_OBS: df = df.sample(MAX_OBS, random_state=42)
    dm = pd.get_dummies(df["depth_num"].astype(str),prefix="D").astype(float)
    dm = dm.drop(columns=[c for c in dm.columns if c=="D_1"],errors='ignore')
    wk = df["week_of_year"].fillna(26).values.astype(float)
    X  = np.column_stack([dm.values,np.sin(2*np.pi*wk/52),np.cos(2*np.pi*wk/52)])
    Y  = df["log_units"].values; sc = df["store_code_uc"].values
    Yd,Xd = within_store(Y,X,sc)
    beta = np.linalg.lstsq(Xd,Yd,rcond=None)[0]
    resid= Yd-Xd@beta; se = cluster_se(Xd,resid,sc)
    cols = list(dm.columns)+["sin_wk","cos_wk"]
    d = {}
    for bn in range(2,8):
        col = f"D_{bn}"
        idx = next((i for i,c in enumerate(cols) if c==col),None)
        if idx is not None: d[bn]=(float(beta[idx]),float(se[idx]))
    results[cat] = d
    print(f"{cat}:", {k:f"b={v[0]:.4f} se={v[1]:.4f}" for k,v in d.items()})

# ── Cross-category pair contrasts ────────────────────────────────────────────
pairs = [("cereal","snacks"),("cereal","coffee"),("snacks","coffee")]
contrast_rows = []
for bn in [6, 7]:
    for c1, c2 in pairs:
        if bn not in results[c1] or bn not in results[c2]: continue
        b1,s1 = results[c1][bn]; b2,s2 = results[c2][bn]
        diff = b1-b2; z_ = diff/np.sqrt(s1**2+s2**2)
        p_, sig_ = sig_label(z_)
        print(f"D{bn} {c1}-{c2}: diff={diff:.4f} z={z_:.3f} p={p_:.3f} {sig_}")
        contrast_rows.append({"bin":f"D{bn}","cat1":c1,"cat2":c2,
                               "b1":round(b1,4),"b2":round(b2,4),
                               "diff":round(diff,4),"z":round(z_,3),
                               "p":round(p_,4),"sig":sig_})

# ── Joint Cochran-Q test for equality at D6 and D7 ──────────────────────────
print("\nJoint equality tests (Cochran's Q):")
for bn in [6, 7]:
    bvec = np.array([results[c][bn][0] for c in CATS if bn in results[c]])
    svec = np.array([results[c][bn][1] for c in CATS if bn in results[c]])
    if len(bvec)<2: continue
    w    = 1/svec**2
    bbar = np.sum(w*bvec)/np.sum(w)
    Q    = np.sum(w*(bvec-bbar)**2)
    df_  = len(bvec)-1
    p_Q  = float(1-stats.chi2.cdf(Q,df_))
    sig_Q= "***" if p_Q<.001 else("**" if p_Q<.01 else("*" if p_Q<.05 else "n.s."))
    print(f"  D{bn}: Cochran Q={Q:.3f} df={df_} p={p_Q:.4f} {sig_Q}")
    for row in contrast_rows:
        if row["bin"]==f"D{bn}": row["cochran_Q"]=round(Q,3); row["cochran_p"]=round(p_Q,4)

# Save
out_cont = pd.DataFrame(contrast_rows)
out_cont.to_csv(f"{OUT_DIR}/cross_cat_wald_v3.csv",index=False)

# Per-cat coefficient table
coef_rows=[]
for cat in CATS:
    r={"category":cat}
    for bn in range(2,8):
        if bn in results[cat]: b,s=results[cat][bn]; r[f"D{bn}_b"]=round(b,4); r[f"D{bn}_se"]=round(s,4)
    coef_rows.append(r)
pd.DataFrame(coef_rows).to_csv(f"{OUT_DIR}/fe_coefs_v3.csv",index=False)

# Monotonicity check
print("\nMonotonicity:")
for cat in CATS:
    blist=[results[cat].get(k,(np.nan,np.nan))[0] for k in range(2,8)]
    violate=[(k+2,k+3) for k in range(len(blist)-1) if blist[k]>blist[k+1]]
    mono="MONOTONE" if not violate else f"NOT monotone: {violate}"
    print(f"  {cat}: {[round(b,3) for b in blist]} → {mono}")

# ── Figure: contrast plot ────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(10,5))
bins=[6,7]; labels=["D6 (40-50%)","D7 (≥50%)"]
CC={"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
x=np.array([0,1]); width=0.25
for i,cat in enumerate(CATS):
    ys=[results[cat].get(bn,(np.nan,np.nan))[0] for bn in bins]
    ses=[results[cat].get(bn,(np.nan,np.nan))[1] for bn in bins]
    ax.bar(x+(i-1)*width,ys,width,color=CC[cat],alpha=0.7,label=cat.capitalize())
    ax.errorbar(x+(i-1)*width,ys,yerr=[1.96*s for s in ses],fmt='none',color='k',capsize=4,lw=1.2)
ax.set_xticks([0,1]); ax.set_xticklabels(labels,fontsize=11)
ax.set_ylabel("FE Coefficient vs D1 (Δ log units)")
ax.set_title("Step 1 v3 — Cross-Category D6/D7 Wald Contrasts\n"
             "(One-way store FE; store-clustered SE; 25k obs subsample; DIAGNOSTIC — not pooled three-way-FE)",
             fontsize=10,fontweight='bold')
ax.legend(); ax.grid(True,alpha=0.3,axis='y'); ax.axhline(0,color='k',lw=0.8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step01_cross_cat_v3.png",dpi=150,bbox_inches='tight')
plt.close()
print("\n✓ Step 1 v3 done")
print(out_cont[["bin","cat1","cat2","diff","z","p","sig"]].to_string(index=False))
