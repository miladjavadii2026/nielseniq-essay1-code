"""Step 6 v2: UPC-exit clearance validation — uses promo_event and W5."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step06_clearance"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
CATS      = ["cereal","snacks","coffee"]

def load_rms(cat, n_rg=2):
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    cols = ["store_code_uc","upc","week_end","log_units","promo_event","W5","discount_depth","depth_bin"]
    avail = pf_.schema_arrow.names
    cols = [c for c in cols if c in avail]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas()
              for i in range(min(n_rg, pf_.num_row_groups))]
    df = pd.concat(frames, ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    return df, cols

all_stats = []

for cat in CATS:
    print(f"\n{'='*45}\n  {cat.upper()}")
    df, cols = load_rms(cat, 2)
    promo_col = "promo_event" if "promo_event" in cols else "W5"
    print(f"  Rows: {len(df):,}  Using promo col: {promo_col}")
    
    terminal_wk = df["week_end"].max()
    print(f"  Terminal week: {terminal_wk.date()}")
    
    # Last observed week per store-UPC
    last_obs = (df.groupby(["store_code_uc","upc"])["week_end"]
                .max().reset_index(name="last_obs_wk"))
    CENSOR_WINDOW = 8
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
    
    # Merge last obs week back
    df_m = df.merge(last_obs, on=["store_code_uc","upc"], how="left")
    df_m["wks_before_last"] = ((df_m["last_obs_wk"] - df_m["week_end"]).dt.days / 7).round().astype(int)
    
    # Focus on EXITING pairs only
    df_exit = df_m[df_m["exit_before_terminal"]==1].copy()
    final8_exit = df_exit[(df_exit["wks_before_last"]>=0)&(df_exit["wks_before_last"]<8)]
    prior8_exit = df_exit[(df_exit["wks_before_last"]>=8)&(df_exit["wks_before_last"]<16)]
    
    pf8e = final8_exit[promo_col].mean() if len(final8_exit)>10 else np.nan
    pp8e = prior8_exit[promo_col].mean() if len(prior8_exit)>10 else np.nan
    diff = pf8e - pp8e if not (np.isnan(pf8e) or np.isnan(pp8e)) else np.nan
    
    # Also compute for all pairs (not just exiting)
    final8_all = df_m[(df_m["wks_before_last"]>=0)&(df_m["wks_before_last"]<8)]
    prior8_all = df_m[(df_m["wks_before_last"]>=8)&(df_m["wks_before_last"]<16)]
    pf8a = final8_all[promo_col].mean(); pp8a = prior8_all[promo_col].mean()
    
    print(f"\n  Promo rate (all pairs)     — final 8wks: {pf8a:.4f}  prior 8wks: {pp8a:.4f}  Δ={pf8a-pp8a:+.4f}")
    print(f"  Promo rate (exiting pairs) — final 8wks: {pf8e:.4f}  prior 8wks: {pp8e:.4f}  Δ={diff:+.4f}")
    
    clearance_signal = "Possible" if (not np.isnan(diff) and diff > 0.05) else "Not supported"
    print(f"  Clearance-inflation signal: {clearance_signal}")
    
    # Deeper: depth_bin distribution near exit
    if "depth_bin" in cols:
        bin_final = final8_exit["depth_bin"].value_counts(normalize=True).sort_index()
        bin_prior = prior8_exit["depth_bin"].value_counts(normalize=True).sort_index()
        print(f"\n  Depth-bin share — final 8wks:\n{bin_final}")
    
    all_stats.append({"category":cat,"terminal_week":str(terminal_wk.date()),
                      "n_pairs":n_total,"n_exit":int(n_exit),"pct_exit":round(pct_exit,1),
                      "n_censored":int(n_censor),"pct_censored":round(pct_cens,1),
                      "promo_final8_exit":round(pf8e,4) if not np.isnan(pf8e) else np.nan,
                      "promo_prior8_exit":round(pp8e,4) if not np.isnan(pp8e) else np.nan,
                      "promo_diff_exit":round(diff,4) if not np.isnan(diff) else np.nan,
                      "promo_final8_all":round(pf8a,4),"promo_prior8_all":round(pp8a,4),
                      "clearance_signal":clearance_signal})

stats = pd.DataFrame(all_stats)
stats.to_csv(f"{OUT_DIR}/clearance_validation.csv", index=False)

# Figure
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle("Step 6 — UPC-Exit Clearance Validation\n"
             "(Promo rate: final 8 wks vs. prior 8 wks for exiting store-UPC pairs; N_RG=2)",
             fontsize=11,fontweight='bold')

cc = {"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for ax,row in zip(axes,all_stats):
    cat = row["category"]
    v_prior = row["promo_prior8_exit"]
    v_final = row["promo_final8_exit"]
    if np.isnan(v_prior): v_prior=0; v_final=0
    bars = ax.bar(["Prior 8wks","Final 8wks"],[v_prior,v_final],
                  color=[cc[cat]+"60",cc[cat]],width=0.5,edgecolor='black',lw=1.5)
    for b,v in zip(bars,[v_prior,v_final]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.3f}",
                ha='center',va='bottom',fontsize=13,fontweight='bold')
    diff = row["promo_diff_exit"]
    sig  = row["clearance_signal"]
    ax.set_title(f"{cat.capitalize()}\nΔ = {diff:+.4f} → {sig}", fontsize=11)
    ax.set_ylabel("Promo Rate (promo_event)")
    ax.set_ylim(0, max(v_prior,v_final)*1.35 + 0.02)
    ax.grid(True,alpha=0.3,axis='y')
    ax.text(0.5,0.93,f"Exiting: {row['n_exit']:,} ({row['pct_exit']}% of pairs)\nCensored (near terminal): {row['pct_censored']}%",
            transform=ax.transAxes,ha='center',va='top',fontsize=9,
            bbox=dict(boxstyle='round',facecolor='lightgray',alpha=0.6))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step06_clearance_validation.png",dpi=150,bbox_inches='tight')
plt.close()
print("\nFigure saved.")
print("\n--- SUMMARY ---")
print(stats[["category","pct_exit","promo_diff_exit","clearance_signal"]].to_string(index=False))
