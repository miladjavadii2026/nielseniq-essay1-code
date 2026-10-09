"""
Step 3: Episode-duration event studies (1-week, 2-week, 3+-week).

Separate promotion episodes by duration class.
Event time 0 = LAST promoted week.
Baseline: tau in {-8,...,-5}; reference: tau=-1.
Ensure no new promotion in post window.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from scipy import stats

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step03_episode_duration"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"

CATS   = ["cereal","snacks","coffee"]
N_RG   = 5
DEPTH  = 0.10
PRE    = 8
POST   = 12

def load_rms(cat, n_rg):
    path = f"{DATA_BASE}/rms_{cat}_analysis.parquet"
    pf_ = pq.ParquetFile(path)
    cols = ["store_code_uc","upc","week_end","discount_depth","log_units"]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas() 
              for i in range(min(n_rg, pf_.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df

def identify_episodes(df, thresh=0.10):
    """Return df with episode_id, episode_duration, is_ep_end."""
    df = df.sort_values(["store_code_uc","upc","week_end"]).copy()
    g = ["store_code_uc","upc"]
    df["is_promo"] = (df["discount_depth"] >= thresh).astype(int)
    df["prv"]      = df.groupby(g)["is_promo"].shift(1, fill_value=0)
    df["nxt"]      = df.groupby(g)["is_promo"].shift(-1, fill_value=0)
    
    df["ep_start"] = (df["is_promo"]==1) & (df["prv"]==0)
    df["ep_end_flag"] = (df["is_promo"]==1) & (df["nxt"]==0)
    
    # Cumulative episode ID within each store-UPC
    df["ep_id"]  = df.groupby(g)["ep_start"].cumsum()
    df["ep_id"]  = df["ep_id"].where(df["is_promo"]==1, other=0)
    
    # Episode duration = count of weeks in each episode
    ep_dur = (df[df["is_promo"]==1]
              .groupby(g + ["ep_id"])
              .size()
              .reset_index(name="ep_duration"))
    df = df.merge(ep_dur, on=g+["ep_id"], how="left")
    df["ep_duration"] = df["ep_duration"].fillna(0).astype(int)
    
    # Duration class
    def dur_class(d):
        if d == 1: return "1wk"
        elif d == 2: return "2wk"
        elif d >= 3: return "3plus_wk"
        else: return "non_promo"
    df["dur_class"] = df["ep_duration"].apply(dur_class)
    
    return df

def build_event_data(df, dur_class_filter, pre=8, post=12, thresh=0.10):
    """Build event-study data for a specific duration class."""
    df = identify_episodes(df, thresh)
    
    # Get episode ends for this duration class
    ep_ends = df[df["ep_end_flag"] & (df["dur_class"]==dur_class_filter)].copy()
    
    g = ["store_code_uc","upc"]
    df["week_idx"] = df.groupby(g).cumcount()
    
    # Map ep_ends to their week_idx
    ep_ends2 = ep_ends.merge(
        df[g+["week_end","week_idx"]],
        on=g+["week_end"], how="left"
    )
    
    records = []
    for _, end_row in ep_ends2.iterrows():
        store = end_row["store_code_uc"]
        upc   = end_row["upc"]
        end_idx = end_row["week_idx"]
        
        su = df[(df["store_code_uc"]==store) & (df["upc"]==upc)].copy()
        su["event_time"] = su["week_idx"] - end_idx
        w = su[(su["event_time"]>=-pre) & (su["event_time"]<=post)].copy()
        
        if len(w) < (pre + post)//2:
            continue
        
        # Clean pre-event window: no episode end before this one
        pre_ep = w[(w["event_time"]<0) & (w["ep_end_flag"])].shape[0]
        if pre_ep > 0:
            continue
        
        # No new promo start in post window
        post_promo = w[(w["event_time"]>0) & (w["is_promo"]==1)].shape[0]
        if post_promo > post * 0.25:  # allow <25%
            continue
        
        records.append(w[["store_code_uc","upc","week_end","log_units","event_time"]])
    
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out = out.drop_duplicates(subset=["store_code_uc","upc","event_time","week_end"])
    return out

def compute_event_means(ep_data):
    """Compute mean and SE of log_units by event_time (simple OLS within-demean)."""
    ep = ep_data.dropna(subset=["log_units","event_time"]).copy()
    ep["event_time"] = ep["event_time"].astype(int)
    
    # Within-demean by store-UPC
    ep["log_units_dm"] = ep["log_units"] - ep.groupby(["store_code_uc","upc"])["log_units"].transform("mean")
    ep["log_units_dm"] += ep.groupby("event_time")["log_units"].transform("mean").groupby(ep["event_time"]).transform("first")
    
    # Simple group means (relative to reference tau=-1)
    means = ep.groupby("event_time")["log_units"].agg(["mean","std","count"]).reset_index()
    means.columns = ["event_time","mean","std","count"]
    means["se"] = means["std"] / np.sqrt(means["count"])
    
    # Normalize to tau=-1
    ref_val = means.loc[means["event_time"]==-1, "mean"].values
    if len(ref_val) > 0:
        means["coef"] = means["mean"] - ref_val[0]
    else:
        means["coef"] = means["mean"] - means["mean"].mean()
    
    means["ci_lo"] = means["coef"] - 1.96 * means["se"]
    means["ci_hi"] = means["coef"] + 1.96 * means["se"]
    return means

REF_TAU = -1

all_results = []

for cat in CATS:
    print(f"\n{'='*55}\n  {cat.upper()}")
    
    df = load_rms(cat, N_RG)
    print(f"  Loaded {len(df):,} rows")
    
    df_ep = identify_episodes(df)
    
    # Count episodes by duration class
    ep_counts = (df_ep[df_ep["ep_end_flag"]]
                 .groupby("dur_class")["ep_id"]
                 .count())
    print(f"  Episode counts: {ep_counts.to_dict()}")
    
    # Process each duration class
    dur_classes = {"1wk": "1-Week Episodes",
                   "2wk": "2-Week Episodes",
                   "3plus_wk": "3+-Week Episodes"}
    
    cat_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Step 3 — Episode Duration Event Studies: {cat.capitalize()}",
                 fontsize=13, fontweight='bold')
    
    colors = {"1wk": "steelblue", "2wk": "darkorange", "3plus_wk": "green"}
    
    for dc, dc_label in dur_classes.items():
        n_ep = ep_counts.get(dc, 0)
        print(f"  {dc_label}: {n_ep} episodes", end="")
        
        ep_data = build_event_data(df, dc)
        
        if len(ep_data) < 50:
            print(f" → SKIP (too few obs)")
            cat_results[dc] = {"n_episodes": n_ep, "n_obs": 0, "status": "insufficient"}
            continue
        
        print(f" | {len(ep_data):,} obs")
        
        means = compute_event_means(ep_data)
        
        ax_idx = list(dur_classes.keys()).index(dc)
        ax = axes[ax_idx]
        
        x  = means["event_time"].values
        y  = means["coef"].values
        lo = means["ci_lo"].values
        hi = means["ci_hi"].values
        
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.axvline(-0.5, color='red', lw=1.5, ls=':', alpha=0.7)
        ax.fill_between(x, lo, hi, alpha=0.2, color=colors[dc])
        ax.plot(x, y, 'o-', ms=4, lw=1.5, color=colors[dc], label=f'n_ep={n_ep:,}')
        
        # Mark baseline and post windows
        ax.axvspan(-8.5,-4.5, alpha=0.08, color='blue')
        ax.axvspan( 0.5,12.5, alpha=0.06, color='orange')
        
        # Annotation: peak effect
        tau0_row = means[means["event_time"]==0]
        if not tau0_row.empty:
            peak = tau0_row["coef"].values[0]
            ax.annotate(f'Peak: {peak:.3f}', xy=(0, peak),
                       xytext=(2, peak+0.05), fontsize=8,
                       arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel('Event time τ (weeks from last promoted week)')
        ax.set_ylabel('Mean log units (demeaned, ref τ=-1)')
        ax.set_title(f'{dc_label}\n(n={n_ep:,} episodes, N={len(ep_data):,} obs)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Record peak, trough, recovery
        tau0 = means[means["event_time"]==0]["coef"].values
        tau1 = means[means["event_time"]==1]["coef"].values
        tau4 = means[means["event_time"]==4]["coef"].values
        
        cat_results[dc] = {
            "category": cat,
            "dur_class": dc,
            "n_episodes": int(n_ep),
            "n_obs": len(ep_data),
            "peak_coef_tau0": round(float(tau0[0]) if len(tau0)>0 else np.nan, 4),
            "post1_coef":     round(float(tau1[0]) if len(tau1)>0 else np.nan, 4),
            "post4_coef":     round(float(tau4[0]) if len(tau4)>0 else np.nan, 4),
            "status": "estimated"
        }
        
        # Save to CSV
        means["category"] = cat
        means["dur_class"] = dc
        means.to_csv(f"{OUT_DIR}/ep_dur_{cat}_{dc}.csv", index=False)
        all_results.extend([{**cat_results[dc], "event_time": row["event_time"],
                             "coef": round(row["coef"],5), "se": round(row["se"],5),
                             "ci_lo": round(row["ci_lo"],5), "ci_hi": round(row["ci_hi"],5),
                             "n_obs_at_tau": int(row["count"])}
                            for _, row in means.iterrows()])
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/step03_episode_duration_{cat}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved.")

# Save summary
if all_results:
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(f"{OUT_DIR}/episode_duration_summary.csv", index=False)
    
    # Print peak effects comparison
    summary = res_df[res_df["event_time"]==0][["category","dur_class",
                                                "n_episodes","peak_coef_tau0"]].copy()
    print(f"\n{'='*55}")
    print("PEAK EFFECTS BY DURATION CLASS:")
    print(summary.to_string(index=False))
else:
    print("No results.")
