"""Step 6 v4 fast: exit-window sensitivity using OLS residuals with store dummies (small n)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step06_clearance"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]
MAX_OBS   = 8000   # very small for speed

def load_small(cat):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","depth_bin","week_of_year"]
    have = [c for c in want if c in pf.schema_arrow.names]
    df   = pf.read_row_group(0, columns=have).to_pandas()
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(int)
    df["wk"]        = pd.Categorical(df["week_of_year"].fillna(26).astype(int)).codes
    # Assign pseudo "last week" per upc (max week_of_year as proxy for exit)
    df["last_wk"] = df.groupby("upc")["week_of_year"].transform("max")
    df["wks_left"]= (df["last_wk"] - df["week_of_year"]).clip(lower=0)
    return df

def fit_ols(sub):
    # Vectorized within-store demean
    for col in ["log_units"] + [f"D{k}" for k in range(2,8)] + ["swk","cwk"]:
        pass  # build below
    dm = pd.get_dummies(sub["depth_num"], prefix="D").rename(
         columns={f"D_{k}":f"D{k}" for k in range(1,8)})
    dm.pop("D1", None)  # D1 = reference
    wk = sub["week_of_year"].fillna(26).values.astype(float)
    dm["swk"] = np.sin(2*np.pi*wk/52); dm["cwk"] = np.cos(2*np.pi*wk/52)
    X    = dm.values.astype(float)
    Y    = sub["log_units"].values.astype(float)
    sto  = sub["store_code_uc"].values
    # Vectorized demean using pandas groupby transform (fast)
    s_ser = pd.Series(sto); dm2 = pd.DataFrame(X)
    Yd = Y - pd.Series(Y).groupby(s_ser).transform("mean").values
    Xd = X - dm2.groupby(s_ser).transform("mean").values
    beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
    resid= Yd - Xd@beta
    S=len(np.unique(sto)); n=len(Yd); p=X.shape[1]
    bread=np.linalg.pinv(Xd.T@Xd); meat=np.zeros((p,p))
    for s in np.unique(sto):
        m=sto==s; sc=Xd[m].T@resid[m]; meat+=np.outer(sc,sc)
    se=np.sqrt(np.diag(S/(S-1)*(n-1)/(n-p)*bread@meat@bread))
    cols=list(dm.columns)
    idx7 = next((i for i,c in enumerate(cols) if c=="D7"), None)
    return (beta[idx7], se[idx7]) if idx7 is not None else (np.nan, np.nan)

all_rows=[]
for cat in CATS:
    print(f"\n{cat.upper()}")
    df = load_small(cat).dropna(subset=["log_units","depth_num"])
    n0 = min(MAX_OBS, len(df))
    df = df.sample(n0, random_state=42).copy() if len(df)>n0 else df.copy()
    for w in [0,4,8,12]:
        sub = df[df["wks_left"]>=w].copy()
        try:
            b7,s7 = fit_ols(sub)
            z7 = b7/s7; sig="***" if abs(z7)>3.29 else("**" if abs(z7)>2.58 else("*" if abs(z7)>1.96 else "n.s."))
            print(f"  excl={w:2d} n={len(sub):,}  D7={b7:.4f}({s7:.4f}) z={z7:.2f}{sig}")
            all_rows.append({"category":cat,"excl":w,"n":len(sub),"D7":round(b7,4),"D7_se":round(s7,4),"z":round(z7,2),"sig":sig})
        except Exception as e:
            print(f"  excl={w} ERROR {e}")
            all_rows.append({"category":cat,"excl":w,"n":len(sub),"D7":np.nan,"D7_se":np.nan,"z":np.nan,"sig":"err"})

out=pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/excl_window_v4.csv",index=False)
fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=False)
CC={"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for ax,cat in zip(axes,CATS):
    r=out[out["category"]==cat]
    ok=r.dropna(subset=["D7"])
    if len(ok):
        ax.errorbar(ok["excl"],ok["D7"],yerr=1.96*ok["D7_se"],fmt='o-',lw=2,ms=9,capsize=5,color=CC[cat])
        for _,row in ok.iterrows():
            ax.text(row["excl"],row["D7"]+0.04,row["sig"],ha='center',fontsize=9,fontweight='bold')
    ax.set_title(f"{cat.capitalize()}\nD7 by exit-exclusion window",fontsize=10,fontweight='bold')
    ax.set_xlabel("Wks excluded before UPC exit"); ax.set_ylabel("D7 FE coef"); ax.grid(True,alpha=0.3)
    ax.set_xticks([0,4,8,12]); ax.set_xticklabels(["Full","Excl\n4wk","Excl\n8wk","Excl\n12wk"])
plt.suptitle("Step 6 v4 — Exit-Window Sensitivity: D7 (store FE, store-clustered SE)",fontsize=11,fontweight='bold',y=1.02)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/step06_excl_window_v4.png",dpi=150,bbox_inches='tight'); plt.close()
print("\n✓ done\n"); print(out.to_string(index=False))
