"""Step 3 v3: Episode duration event studies — minimal approach."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step03_episode_duration"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS = ["cereal","snacks","coffee"]
N_RG = 2; DEPTH = 0.10; PRE = 8; POST = 12

def load_rms(cat, n_rg=2):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    cols = ["store_code_uc","upc","week_end","discount_depth","log_units"]
    df = pd.concat([pf_.read_row_group(i, columns=cols).to_pandas()
                    for i in range(min(n_rg, pf_.num_row_groups))], ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df

def build_all_events(df, thresh=0.10, pre=8, post=12):
    """Build event_study data for all episodes, tagged by duration class."""
    df = df.sort_values(["store_code_uc","upc","week_end"]).copy().reset_index(drop=True)
    g = ["store_code_uc","upc"]
    df["is_promo"] = (df["discount_depth"] >= thresh).astype(int)
    df["nxt"] = df.groupby(g)["is_promo"].shift(-1, fill_value=0)
    df["prv"] = df.groupby(g)["is_promo"].shift(1, fill_value=0)
    df["ep_start"] = ((df["is_promo"]==1)&(df["prv"]==0)).astype(int)
    df["ep_end"]   = ((df["is_promo"]==1)&(df["nxt"]==0)).astype(int)
    df["ep_id"]    = df.groupby(g)["ep_start"].cumsum()
    df["ridx"]     = df.groupby(g).cumcount()
    
    # Episode durations
    ep_dur = (df[df["is_promo"]==1].groupby(g+["ep_id"])
              .size().reset_index(name="ep_dur"))
    df = df.merge(ep_dur, on=g+["ep_id"], how="left")
    df["ep_dur"] = df["ep_dur"].fillna(0).astype(int)
    
    def dc(d):
        if d==1: return "1wk"
        if d==2: return "2wk"
        if d>=3: return "3plus"
        return "nonpromo"
    df["dur_class"] = df["ep_dur"].apply(dc)
    
    # Find episode ends
    ends = df[df["ep_end"]==1].copy()
    ends = ends[ends["dur_class"].isin(["1wk","2wk","3plus"])].copy()
    ends["ep_key"] = ends["store_code_uc"]+"_"+ends["upc"]+"_"+ends["ep_id"].astype(str)
    
    print(f"  Episode ends: {len(ends)} | by class: {ends['dur_class'].value_counts().to_dict()}")
    
    # Merge ends with obs
    obs = df[["store_code_uc","upc","ridx","log_units","is_promo","ep_id"]].copy()
    
    ev_parts = []
    for dc_val in ["1wk","2wk","3plus"]:
        e = ends[ends["dur_class"]==dc_val][["store_code_uc","upc","ridx","ep_key","dur_class"]].copy()
        e = e.rename(columns={"ridx":"end_ridx"})
        m = obs.merge(e, on=["store_code_uc","upc"], how="inner")
        m["event_time"] = m["ridx"] - m["end_ridx"]
        m = m[(m["event_time"]>=-pre)&(m["event_time"]<=post)]
        ev_parts.append(m)
    
    ev = pd.concat(ev_parts, ignore_index=True) if ev_parts else pd.DataFrame()
    if len(ev) == 0:
        return pd.DataFrame()
    
    ev = ev.dropna(subset=["log_units"])
    ev = ev.drop_duplicates(subset=["store_code_uc","upc","ep_key","event_time"])
    return ev

all_rows = []

for cat in CATS:
    print(f"\n{'='*55}\n  {cat.upper()}")
    df = load_rms(cat, N_RG)
    print(f"  {len(df):,} rows")
    
    ev = build_all_events(df)
    if len(ev) == 0:
        print("  No episodes found."); continue
    
    print(f"  Total event obs: {len(ev):,}")
    
    # Within-demean by store-UPC
    ev["lu_dm"] = ev["log_units"] - ev.groupby(["store_code_uc","upc"])["log_units"].transform("mean")
    
    # Count episodes per class
    ep_n = ev[ev["event_time"]==0].groupby("dur_class")["ep_key"].nunique()
    
    dur_classes = [("1wk","1-Week","steelblue"),
                   ("2wk","2-Week","darkorange"),
                   ("3plus","3+-Week","forestgreen")]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Step 3 — Duration Event Studies: {cat.capitalize()}",
                 fontsize=12, fontweight='bold')
    
    for ax_i, (dc_val, dc_label, col) in enumerate(dur_classes):
        sub = ev[ev["dur_class"]==dc_val]
        n_ep = int(ep_n.get(dc_val, 0))
        ax = axes[ax_i]
        
        if len(sub) < 30 or n_ep == 0:
            ax.text(0.5,0.5,f'n_ep={n_ep}\n(insufficient)',
                    ha='center',va='center',transform=ax.transAxes,fontsize=11)
            ax.set_title(f'{dc_label}-Week Episodes')
            continue
        
        gp = sub.groupby("event_time")["lu_dm"].agg(["mean","std","count"]).reset_index()
        gp.columns = ["event_time","coef","std","n"]
        gp["se"] = gp["std"] / np.sqrt(gp["n"])
        ref = gp.loc[gp["event_time"]==-1,"coef"].values
        ref_v = float(ref[0]) if len(ref)>0 else 0.0
        gp["coef"] -= ref_v
        gp["ci_lo"] = gp["coef"] - 1.96*gp["se"]
        gp["ci_hi"] = gp["coef"] + 1.96*gp["se"]
        
        ax.axhline(0,color='gray',lw=0.8,ls='--')
        ax.axvline(-0.5,color='red',lw=1.2,ls=':',alpha=0.8)
        ax.axvspan(-8.5,-4.5,alpha=0.08,color='royalblue')
        ax.axvspan(-4.5,0.5,alpha=0.08,color='green')
        ax.axvspan(0.5,12.5,alpha=0.06,color='orange')
        ax.fill_between(gp["event_time"],gp["ci_lo"],gp["ci_hi"],alpha=0.25,color=col)
        ax.plot(gp["event_time"],gp["coef"],'o-',ms=5,lw=2,color=col,label=f'n_ep={n_ep}')
        
        p0 = gp.loc[gp["event_time"]==0,"coef"].values
        if len(p0):
            ax.annotate(f'τ=0:{p0[0]:.3f}',xy=(0,p0[0]),
                       xytext=(2,p0[0]+0.05),fontsize=8,
                       arrowprops=dict(arrowstyle='->',lw=0.8))
        
        ax.set_xlabel('Event time τ (weeks from last promo week)')
        ax.set_ylabel('Within-demeaned log units (ref=τ=-1)')
        ax.set_title(f'{dc_label}-Week Episodes\n(n_ep={n_ep}, N={len(sub):,})')
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        
        gp["category"]=cat; gp["dur_class"]=dc_val; gp["n_episodes"]=n_ep
        gp.to_csv(f"{OUT_DIR}/ep_dur_{cat}_{dc_val}.csv",index=False)
        for _,r in gp.iterrows():
            all_rows.append({"category":cat,"dur_class":dc_val,"n_episodes":n_ep,
                             "event_time":int(r["event_time"]),"coef":round(r["coef"],5),
                             "se":round(r["se"],5),"ci_lo":round(r["ci_lo"],5),
                             "ci_hi":round(r["ci_hi"],5),"n_obs_at_tau":int(r["n"])})
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/step03_episode_duration_{cat}.png",dpi=150,bbox_inches='tight')
    plt.close(); print(f"  Figure saved.")

if all_rows:
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(f"{OUT_DIR}/episode_duration_all.csv",index=False)
    peak = out_df[out_df["event_time"]==0][["category","dur_class","n_episodes","coef"]].copy()
    print(f"\n{'='*55}\nPEAK EFFECT (tau=0):")
    print(peak.to_string(index=False))
