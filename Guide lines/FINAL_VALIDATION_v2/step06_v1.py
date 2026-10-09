"""Step 6: UPC-exit clearance validation — end-of-panel censoring check."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step06_clearance"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = {"cereal":1344,"snacks":1323,"coffee":1463}

def load_rms(cat, n_rg=2):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    cols = ["store_code_uc","upc","week_end","log_units","is_promo","promo_depth"]
    avail = pf_.schema_arrow.names
    cols = [c for c in cols if c in avail]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas()
              for i in range(min(n_rg, pf_.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    return df

all_stats = []

for cat in CATS:
    print(f"\n{'='*45}\n  {cat.upper()}")
    df = load_rms(cat, 2)
    print(f"  Rows: {len(df):,}  UPCs: {df['upc'].nunique()}")
    
    # Terminal week = max week in the panel
    terminal_wk = df["week_end"].max()
    print(f"  Terminal week: {terminal_wk.date()}")
    
    # For each store-UPC pair, find their last observed week
    last_obs = (df.groupby(["store_code_uc","upc"])["week_end"]
                .max().reset_index(name="last_obs_wk"))
    
    # Flag pairs whose last obs is within 8 weeks of terminal (= censored)
    CENSOR_WINDOW = 8  # weeks
    censor_cutoff = terminal_wk - pd.Timedelta(weeks=CENSOR_WINDOW)
    last_obs["near_terminal"] = (last_obs["last_obs_wk"] >= censor_cutoff).astype(int)
    last_obs["exit_before_terminal"] = (last_obs["last_obs_wk"] < censor_cutoff).astype(int)
    
    n_total = len(last_obs)
    n_exit  = last_obs["exit_before_terminal"].sum()
    n_censor= last_obs["near_terminal"].sum()
    pct_exit = 100*n_exit/n_total
    pct_cens = 100*n_censor/n_total
    
    print(f"  Store-UPC pairs: {n_total:,}")
    print(f"  Exit before terminal (>8wks gap): {n_exit:,} ({pct_exit:.1f}%)")
    print(f"  Censored (near terminal): {n_censor:,} ({pct_cens:.1f}%)")
    
    # Check if exiting pairs had higher promo rates just before exit
    df_m = df.merge(last_obs, on=["store_code_uc","upc"], how="left")
    
    # Final 8 weeks vs prior 8 weeks
    df_m["wks_before_last"] = ((df_m["last_obs_wk"] - df_m["week_end"]).dt.days / 7).round().astype(int)
    
    final8  = df_m[(df_m["wks_before_last"]>=0)&(df_m["wks_before_last"]<8)]
    prior8  = df_m[(df_m["wks_before_last"]>=8)&(df_m["wks_before_last"]<16)]
    
    promo_final = final8["is_promo"].mean() if len(final8)>0 else np.nan
    promo_prior = prior8["is_promo"].mean() if len(prior8)>0 else np.nan
    
    # By exit vs. censored
    exiting_pairs = last_obs[last_obs["exit_before_terminal"]==1][["store_code_uc","upc"]]
    df_exit = df_m.merge(exiting_pairs, on=["store_code_uc","upc"])
    final8_exit = df_exit[(df_exit["wks_before_last"]>=0)&(df_exit["wks_before_last"]<8)]
    prior8_exit = df_exit[(df_exit["wks_before_last"]>=8)&(df_exit["wks_before_last"]<16)]
    
    pf8e = final8_exit["is_promo"].mean() if len(final8_exit)>0 else np.nan
    pp8e = prior8_exit["is_promo"].mean() if len(prior8_exit)>0 else np.nan
    diff = pf8e - pp8e if not (np.isnan(pf8e) or np.isnan(pp8e)) else np.nan
    
    print(f"\n  Promo rate (all pairs) — final 8 wks: {promo_final:.4f}  prior 8 wks: {promo_prior:.4f}")
    print(f"  Promo rate (exiting pairs) — final 8 wks: {pf8e:.4f}  prior 8 wks: {pp8e:.4f}")
    print(f"  Difference (final-prior) for exiting: {diff:+.4f}")
    
    clearance_signal = "Possible" if (diff is not np.nan and diff > 0.05) else "Not supported"
    print(f"  Clearance-inflation signal: {clearance_signal}")
    
    row = {"category":cat, "terminal_week":str(terminal_wk.date()),
           "n_pairs":n_total,"n_exit":int(n_exit),"pct_exit":round(pct_exit,1),
           "n_censored":int(n_censor),"pct_censored":round(pct_cens,1),
           "promo_final8_all":round(promo_final,4),"promo_prior8_all":round(promo_prior,4),
           "promo_final8_exit":round(pf8e,4),"promo_prior8_exit":round(pp8e,4),
           "promo_diff_exit":round(diff,4) if not np.isnan(diff) else np.nan,
           "clearance_signal":clearance_signal}
    all_stats.append(row)

stats = pd.DataFrame(all_stats)
stats.to_csv(f"{OUT_DIR}/clearance_validation.csv", index=False)
print("\n--- SUMMARY ---")
print(stats[["category","pct_exit","promo_diff_exit","clearance_signal"]].to_string(index=False))

# Figure
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle("Step 6 — UPC-Exit Clearance Validation\n"
             "(Compare promo rate: final 8 wks vs. prior 8 wks for exiting store-UPC pairs)",
             fontsize=11,fontweight='bold')

cc = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for ax,row in zip(axes,all_stats):
    cat = row["category"]
    vals = [row["promo_prior8_exit"], row["promo_final8_exit"]]
    bars = ax.bar(["Prior 8wks","Final 8wks"], vals,
                  color=[cc[cat]+"80",cc[cat]], width=0.5, edgecolor='black')
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.3f}",
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    diff = row["promo_diff_exit"]
    sig  = row["clearance_signal"]
    ax.set_title(f"{cat.capitalize()}\nΔ = {diff:+.4f} → {sig}",fontsize=11)
    ax.set_ylabel("Promo Rate (exiting pairs)")
    ax.set_ylim(0, max(vals)*1.3)
    ax.grid(True,alpha=0.3,axis='y')
    # Annotation
    ax.text(0.5, 0.95, f"Exit pairs: {row['n_exit']:,} ({row['pct_exit']}%)\nCensored: {row['n_censored']:,} ({row['pct_censored']}%)",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step06_clearance_validation.png",dpi=150,bbox_inches='tight')
plt.close()
print("\nFigure saved.")
