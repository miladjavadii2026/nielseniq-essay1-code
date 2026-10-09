"""Step 3 v2: Episode duration event studies — vectorized."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step03_episode_duration"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS = ["cereal","snacks","coffee"]
N_RG = 3; DEPTH = 0.10; PRE = 8; POST = 12

def load_rms(cat, n_rg=3):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    cols = ["store_code_uc","upc","week_end","discount_depth","log_units"]
    df = pd.concat([pf_.read_row_group(i, columns=cols).to_pandas()
                    for i in range(min(n_rg, pf_.num_row_groups))], ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df

def vectorized_episode_event_study(df, thresh=0.10, pre=8, post=12):
    """
    Fully vectorized episode detection and event-time assignment.
    Returns: event_df with [store_code_uc, upc, event_time, log_units, dur_class, ep_key]
    """
    df = df.sort_values(["store_code_uc","upc","week_end"]).copy().reset_index(drop=True)
    g = ["store_code_uc","upc"]
    df["is_promo"] = (df["discount_depth"] >= thresh).astype(int)
    
    # Lag/lead for episode boundary detection
    df["prv"] = df.groupby(g)["is_promo"].shift(1, fill_value=0)
    df["nxt"] = df.groupby(g)["is_promo"].shift(-1, fill_value=0)
    df["ep_start"] = ((df["is_promo"]==1) & (df["prv"]==0)).astype(int)
    df["ep_end"]   = ((df["is_promo"]==1) & (df["nxt"]==0)).astype(int)
    
    # Assign episode ID
    df["ep_id_raw"] = df.groupby(g)["ep_start"].cumsum()
    df["ep_id"] = df["ep_id_raw"].where(df["is_promo"]==1, other=0)
    
    # Episode duration
    ep_dur = (df[df["is_promo"]==1]
              .groupby(g+["ep_id_raw"]).size()
              .reset_index(name="ep_dur"))
    df = df.merge(ep_dur, on=g+["ep_id_raw"], how="left")
    df["ep_dur"] = df["ep_dur"].fillna(0).astype(int)
    
    def dc(d):
        if d==1: return "1wk"
        if d==2: return "2wk"
        if d>=3: return "3plus"
        return None
    df["dur_class"] = df["ep_dur"].apply(dc)
    
    # Sequential row index within store-UPC
    df["row_idx"] = df.groupby(g).cumcount()
    
    # Get episode ends with their row_idx
    ends = df[df["ep_end"]==1].copy()
    ends = ends[ends["dur_class"].notna()].copy()
    ends["ep_key"] = (ends["store_code_uc"] + "_" + ends["upc"] + "_" +
                      ends["ep_id_raw"].astype(str))
    
    # Merge ends with ALL observations (within the same store-UPC)
    # Then compute event_time = row_idx_obs - row_idx_end
    merged = df.merge(
        ends[["store_code_uc","upc","row_idx","dur_class","ep_key"]].rename(
            columns={"row_idx":"end_row"}),
        on=["store_code_uc","upc"], how="inner"
    )
    merged["event_time"] = merged["row_idx"] - merged["end_row"]
    
    # Filter to window
    merged = merged[(merged["event_time"] >= -pre) & (merged["event_time"] <= post)]
    
    # Contamination filter: drop episode if there's any other ep_end in pre window
    # Flag rows that ARE an ep_end (not the current one) in pre window
    ep_other_ends_pre = (
        merged[(merged["event_time"] < 0) & (merged["ep_end"]==1) & 
               (merged["ep_id_raw"] != merged["end_row"]-0)]  # simplified check
        .groupby("ep_key")["event_time"].count()
    )
    # Actually use simpler: drop ep_key if any is_promo start in pre window (from different ep)
    other_ep_in_pre = (
        merged[(merged["event_time"] < -1) & (merged["ep_start"]==1)]
        .groupby("ep_key")["event_time"].count()
    )
    bad_keys = set(other_ep_in_pre[other_ep_in_pre > 0].index)
    merged = merged[~merged["ep_key"].isin(bad_keys)]
    
    # Drop rows with NaN log_units
    merged = merged.dropna(subset=["log_units"])
    merged = merged.drop_duplicates(subset=["store_code_uc","upc","ep_key","event_time"])
    
    return merged

all_rows = []

for cat in CATS:
    print(f"\n{'='*55}\n  {cat.upper()}")
    df = load_rms(cat, N_RG)
    print(f"  {len(df):,} rows loaded")
    
    ev = vectorized_episode_event_study(df)
    
    dc_counts = ev[ev["event_time"]==0].groupby("dur_class")["ep_key"].nunique()
    print(f"  Episodes by class: {dc_counts.to_dict()}")
    
    # ── Demean within store-UPC (within estimator) ──
    ev["lu_dm"] = ev["log_units"] - ev.groupby(["store_code_uc","upc"])["log_units"].transform("mean")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Step 3 — Episode Duration Event Studies: {cat.capitalize()}\n"
                 f"(N_RG={N_RG} sandbox; event time 0 = last promoted week)",
                 fontsize=12, fontweight='bold')
    
    dur_classes = [("1wk","1-Week Episodes","steelblue"),
                   ("2wk","2-Week Episodes","darkorange"),
                   ("3plus","3+-Week Episodes","forestgreen")]
    
    for ax_i, (dc, dc_label, col) in enumerate(dur_classes):
        sub = ev[ev["dur_class"]==dc].copy()
        n_ep = dc_counts.get(dc, 0)
        
        if len(sub) < 50:
            axes[ax_i].text(0.5, 0.5, f'Insufficient data\n(n_ep={n_ep})',
                            ha='center', va='center', transform=axes[ax_i].transAxes)
            axes[ax_i].set_title(f'{dc_label}\n(n_ep={n_ep})')
            continue
        
        # Mean demeaned log_units by event_time
        gp = sub.groupby("event_time")["lu_dm"].agg(["mean","std","count"]).reset_index()
        gp.columns = ["event_time","coef","std","n"]
        gp["se"] = gp["std"] / np.sqrt(gp["n"])
        
        # Normalize to tau=-1
        ref_row = gp[gp["event_time"]==-1]
        ref_val = ref_row["coef"].values[0] if len(ref_row)>0 else 0.0
        gp["coef"] -= ref_val
        gp["ci_lo"] = gp["coef"] - 1.96*gp["se"]
        gp["ci_hi"] = gp["coef"] + 1.96*gp["se"]
        
        ax = axes[ax_i]
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.axvline(-0.5, color='red', lw=1.2, ls=':', alpha=0.8, label='Promo end')
        ax.axvspan(-8.5,-4.5, alpha=0.10, color='royalblue')
        ax.axvspan(-4.5, 0.5, alpha=0.10, color='green')
        ax.axvspan( 0.5,12.5, alpha=0.07, color='orange')
        ax.fill_between(gp["event_time"], gp["ci_lo"], gp["ci_hi"],
                        alpha=0.25, color=col)
        ax.plot(gp["event_time"], gp["coef"], 'o-', ms=5, lw=2,
                color=col, label=f'n_ep={n_ep:,}')
        
        peak = gp.loc[gp["event_time"]==0,"coef"].values
        if len(peak)>0:
            ax.annotate(f'τ=0:\n{peak[0]:.3f}',
                       xy=(0,peak[0]), xytext=(2,peak[0]+abs(peak[0])*0.15+0.03),
                       fontsize=8, arrowprops=dict(arrowstyle='->',lw=0.8))
        
        ax.set_xlabel('Event time τ')
        ax.set_ylabel('Within-demeaned log units\n(ref = τ=-1)')
        ax.set_title(f'{dc_label}\n(n_ep={n_ep:,}, N={len(sub):,} obs)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Save CSV
        gp["category"] = cat; gp["dur_class"] = dc; gp["n_episodes"] = n_ep
        gp.to_csv(f"{OUT_DIR}/ep_dur_{cat}_{dc}.csv", index=False)
        
        for _, row in gp.iterrows():
            all_rows.append({"category":cat, "dur_class":dc, "n_episodes":n_ep,
                             "event_time":int(row["event_time"]),
                             "coef":round(row["coef"],5),
                             "se":round(row["se"],5),
                             "ci_lo":round(row["ci_lo"],5),
                             "ci_hi":round(row["ci_hi"],5),
                             "n_obs_at_tau":int(row["n"])})
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/step03_episode_duration_{cat}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved.")

if all_rows:
    pd.DataFrame(all_rows).to_csv(f"{OUT_DIR}/episode_duration_all.csv", index=False)
    # Summary: peak effect by category and class
    summ = pd.DataFrame(all_rows)
    peak = summ[summ["event_time"]==0][["category","dur_class","n_episodes","coef"]].copy()
    print(f"\n{'='*55}")
    print("PEAK EFFECT (tau=0) BY DURATION CLASS:")
    print(peak.to_string(index=False))
