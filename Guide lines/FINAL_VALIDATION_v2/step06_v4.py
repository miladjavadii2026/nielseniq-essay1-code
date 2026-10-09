"""Step 6 v4: Exit-window sensitivity — D7 FE coef at 0/4/8/12 week exclusion windows."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')

DATA_BASE  = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR    = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step06_clearance"
FIG_DIR    = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS       = ["cereal","snacks","coffee"]
W_EXCL_LIST= [0, 4, 8, 12]
MAX_OBS    = 20000

def load_cat(cat):
    pf   = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    want = ["store_code_uc","upc","log_units","depth_bin","week_of_year"]
    have = [c for c in want if c in pf.schema_arrow.names]
    frames = [pf.read_row_group(i, columns=have).to_pandas() for i in range(min(1, pf.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["depth_num"] = df["depth_bin"].astype(str).str[1:].astype(int)
    return df

def fit_fe(sub):
    dm = pd.get_dummies(sub["depth_num"], prefix="D", drop_first=False).astype(float)
    dm = dm.drop(columns=[c for c in dm.columns if c=="D_1"], errors='ignore')
    wk = sub["week_of_year"].fillna(26).values.astype(float)
    X  = np.column_stack([dm.values, np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52)])
    Y  = sub["log_units"].values
    sto= sub["store_code_uc"].values
    # Within-store demean
    Xd = X.copy().astype(float); Yd = Y.copy().astype(float)
    for s in np.unique(sto):
        m = sto==s
        if m.sum()<2: continue
        Yd[m] -= Yd[m].mean(); Xd[m] -= Xd[m].mean(axis=0)
    beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
    resid= Yd - Xd@beta
    # Cluster SE
    S   = len(np.unique(sto)); n=len(Yd); p=X.shape[1]
    bread= np.linalg.pinv(Xd.T@Xd)
    meat = np.zeros((p,p))
    for s in np.unique(sto):
        m=sto==s; sc=Xd[m].T@resid[m]; meat+=np.outer(sc,sc)
    se = np.sqrt(np.diag(S/(S-1)*(n-1)/(n-p)*bread@meat@bread))
    cols= list(dm.columns)+["sin_wk","cos_wk"]
    return dict(zip(cols, zip(beta, se)))

all_rows = []
for cat in CATS:
    print(f"\n{cat.upper()}")
    df = load_cat(cat).dropna(subset=["log_units","depth_num"]).copy()
    upc_last = df.groupby("upc")["week_of_year"].max()
    df["last_wk"] = df["upc"].map(upc_last)
    df["wks_left"] = df["last_wk"] - df["week_of_year"]
    for w in W_EXCL_LIST:
        sub = df[df["wks_left"]>=w].copy()
        nd  = len(df)-len(sub)
        if len(sub)>MAX_OBS: sub = sub.sample(MAX_OBS, random_state=42)
        try:
            res = fit_fe(sub)
            b7,s7 = res.get("D_7",(np.nan,np.nan))
            z7 = b7/s7 if s7 and s7>0 else np.nan
            sig= "***" if abs(z7)>3.29 else ("**" if abs(z7)>2.58 else ("*" if abs(z7)>1.96 else "n.s."))
            print(f"  excl={w:2d}wk n={len(sub):,} drop={nd:,}  D7={b7:.4f}({s7:.4f}) z={z7:.2f}{sig}")
            all_rows.append({"category":cat,"excl":w,"n":len(sub),"D7":round(b7,4),"D7_se":round(s7,4),"z":round(z7,2),"sig":sig})
        except Exception as e:
            print(f"  excl={w}wk ERROR {e}")
            all_rows.append({"category":cat,"excl":w,"n":len(sub),"D7":np.nan,"D7_se":np.nan,"z":np.nan,"sig":"err"})

out = pd.DataFrame(all_rows)
out.to_csv(f"{OUT_DIR}/excl_window_v4.csv", index=False)

fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=False)
for ax,cat in zip(axes,CATS):
    r=out[out["category"]==cat].copy()
    ax.errorbar(r["excl"],r["D7"],yerr=1.96*r["D7_se"],fmt='o-',lw=2,ms=9,capsize=6,
                color={"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}[cat])
    for _,row in r.iterrows():
        ax.text(row["excl"],row["D7"]+0.04,row["sig"],ha='center',fontsize=9,fontweight='bold')
    ax.set_title(f"{cat.capitalize()}\nD7 coef by exclusion window",fontsize=10,fontweight='bold')
    ax.set_xlabel("Weeks excluded pre-exit"); ax.set_ylabel("D7 FE coef"); ax.grid(True,alpha=0.3)
    ax.set_xticks([0,4,8,12]); ax.set_xticklabels(["Full","Excl\n4wk","Excl\n8wk","Excl\n12wk"])
plt.suptitle("Step 6 v4 — Exit-Window Sensitivity: D7 Coefficient (store FE, clustered SE)",fontsize=11,fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step06_excl_window_v4.png",dpi=150,bbox_inches='tight')
plt.close()
print("\n✓ done")
print(out.to_string(index=False))
