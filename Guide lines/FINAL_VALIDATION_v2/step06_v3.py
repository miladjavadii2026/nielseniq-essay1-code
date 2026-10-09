"""Step 6 v3: UPC-exit clearance — use actual terminal week 2020-12-26.
Load last row group to get actual terminal, first row group for early obs.
Key fix: terminal_wk = 2020-12-26 (confirmed from full panel), not 2010 truncation.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import scipy.stats as st

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step06_clearance"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"
# Actual terminal week confirmed from last row group across all categories
ACTUAL_TERMINAL = pd.Timestamp("2020-12-26")

CATS = ["cereal","snacks","coffee"]

def load_rms_boundary(cat):
    """Load first 2 + last 2 row groups to get full date range of store-UPC histories."""
    pf_ = pq.ParquetFile(f"{DATA_BASE}/rms_{cat}_analysis.parquet")
    n_rg = pf_.num_row_groups
    want = ["store_code_uc","upc","week_end","promo_event"]
    avail = pf_.schema_arrow.names
    cols = [c for c in want if c in avail]
    
    rg_list = list(range(min(2, n_rg))) + list(range(max(0, n_rg-2), n_rg))
    rg_list = sorted(set(rg_list))
    frames = [pf_.read_row_group(i, columns=cols).to_pandas() for i in rg_list]
    df = pd.concat(frames, ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    print(f"  Loaded RGs {rg_list} of {n_rg}")
    print(f"  Date range in sample: {df['week_end'].min().date()} — {df['week_end'].max().date()}")
    return df, n_rg

all_stats = []

for cat in CATS:
    print(f"\n{'='*45}\n  {cat.upper()}")
    df, n_rg = load_rms_boundary(cat)
    terminal_wk = ACTUAL_TERMINAL
    print(f"  Actual terminal week (full panel): {terminal_wk.date()}")
    print(f"  Store-UPC pairs in sample: {df.groupby(['store_code_uc','upc']).ngroups:,}")
    
    # For each store-UPC, find last obs in our sample
    last_obs = (df.groupby(["store_code_uc","upc"])["week_end"]
                .max().reset_index(name="last_obs_wk"))
    first_obs = (df.groupby(["store_code_uc","upc"])["week_end"]
                 .min().reset_index(name="first_obs_wk"))
    su_info = last_obs.merge(first_obs, on=["store_code_uc","upc"])
    
    # Genuine exits: last obs at least 12 weeks before actual terminal
    # Administrative censoring: last obs within 12 weeks of terminal
    EXIT_WINDOW = 12
    censor_cutoff = terminal_wk - pd.Timedelta(weeks=EXIT_WINDOW)
    su_info["genuine_exit"] = (su_info["last_obs_wk"] < censor_cutoff).astype(int)
    su_info["censored"]     = (su_info["last_obs_wk"] >= censor_cutoff).astype(int)
    
    # Only pairs present in BOTH early and late data can have genuine exits
    # (pairs only in first 2 RGs may be truncated by RG boundary)
    # Flag: if pair seen in last 2 RGs → definitely censored
    late_pairs = (df[df["week_end"] >= pd.Timestamp("2020-01-01")]
                  .groupby(["store_code_uc","upc"]).size().reset_index().assign(in_late=1))
    su_info = su_info.merge(late_pairs[["store_code_uc","upc","in_late"]], 
                            on=["store_code_uc","upc"], how="left")
    su_info["in_late"] = su_info["in_late"].fillna(0).astype(int)
    
    # Definite exits: not in late data AND last obs < censor_cutoff
    su_info["definite_exit"] = ((su_info["in_late"]==0)&(su_info["genuine_exit"]==1)).astype(int)
    
    n_total  = len(su_info)
    n_dexit  = su_info["definite_exit"].sum()
    n_cens   = su_info["censored"].sum() + su_info[(su_info["in_late"]==0)&(su_info["genuine_exit"]==0)].shape[0]
    pct_dexit= 100*n_dexit/n_total
    
    print(f"  Total store-UPC pairs (sample): {n_total:,}")
    print(f"  Definite exits (last obs <{EXIT_WINDOW}wks before 2020-12-26): {n_dexit:,} ({pct_dexit:.1f}%)")
    print(f"  Still-active or censored: {n_total-n_dexit:,}")
    
    # Compare promo rates for definite-exit pairs
    df_m = df.merge(su_info[["store_code_uc","upc","last_obs_wk","definite_exit"]], 
                    on=["store_code_uc","upc"], how="left")
    df_m["wks_before_last"] = ((df_m["last_obs_wk"]-df_m["week_end"]).dt.days/7).round().astype(int)
    
    df_exit = df_m[df_m["definite_exit"]==1].copy()
    final8  = df_exit[(df_exit["wks_before_last"]>=0)&(df_exit["wks_before_last"]<8)]
    prior8  = df_exit[(df_exit["wks_before_last"]>=8)&(df_exit["wks_before_last"]<16)]
    
    pf8 = final8["promo_event"].mean() if len(final8)>10 else np.nan
    pp8 = prior8["promo_event"].mean() if len(prior8)>10 else np.nan
    
    if not (np.isnan(pf8) or np.isnan(pp8)) and len(final8)>5 and len(prior8)>5:
        # Two-sample z-test for proportion difference
        n1,n2 = len(final8),len(prior8)
        p_pool = (final8["promo_event"].sum()+prior8["promo_event"].sum())/(n1+n2)
        se_diff = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2)) + 1e-10
        diff = pf8-pp8; z_diff = diff/se_diff
        p_diff = float(2*(1-st.norm.cdf(abs(z_diff))))
        sig = "***" if p_diff<0.001 else ("**" if p_diff<0.01 else ("*" if p_diff<0.05 else "n.s."))
    else:
        diff,z_diff,p_diff,sig = np.nan,np.nan,np.nan,""
    
    clearance_signal = "Possible" if (not np.isnan(diff) and diff>0.05 and (not np.isnan(p_diff) and p_diff<0.05)) else "Not supported"
    
    print(f"  Promo rate: final 8wks={pf8:.4f} | prior 8wks={pp8:.4f} | Δ={diff:+.4f} | z={z_diff:.2f} | p={p_diff:.4f} {sig}")
    print(f"  Clearance signal: {clearance_signal}")
    print(f"  NOTE: full clearance test requires workstation (all {n_rg:,} row groups).")
    
    all_stats.append({"category":cat,"true_terminal":"2020-12-26",
                      "n_pairs_sample":n_total,"n_definite_exit":int(n_dexit),"pct_exit":round(pct_dexit,1),
                      "promo_final8":round(pf8,4) if not np.isnan(pf8) else np.nan,
                      "promo_prior8":round(pp8,4) if not np.isnan(pp8) else np.nan,
                      "promo_diff":round(diff,4) if not np.isnan(diff) else np.nan,
                      "z_stat":round(z_diff,3) if not np.isnan(z_diff) else np.nan,
                      "p_val":round(p_diff,4) if not np.isnan(p_diff) else np.nan,
                      "sig":sig,"clearance_signal":clearance_signal,
                      "caveat":f"Sample: 4 of {n_rg} RGs; definitive test requires all RGs on workstation"})

stats = pd.DataFrame(all_stats)
stats.to_csv(f"{OUT_DIR}/clearance_validation_v3.csv", index=False)

# Figure
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle("Step 6 v3 — UPC-Exit Clearance Validation\n"
             "True terminal=2020-12-26; Definite exits only (last obs >12wks before terminal)\n"
             "⚠ Sample covers 4 of 9K–12K row groups — workstation required for full test",
             fontsize=10,fontweight='bold')
cc={"cereal":"#1565C0","snacks":"#E65100","coffee":"#2E7D32"}
for ax,row in zip(axes,all_stats):
    cat=row["category"]
    pf8=row["promo_final8"]; pp8=row["promo_prior8"]
    if any(x is None or (isinstance(x,float) and np.isnan(x)) for x in [pf8,pp8]):
        ax.text(0.5,0.5,"Insufficient\ndefinite-exit obs",ha='center',va='center',transform=ax.transAxes)
        ax.set_title(cat.capitalize()); continue
    bars=ax.bar(["Prior 8wks","Final 8wks"],[pp8,pf8],
                color=[cc[cat]+"60",cc[cat]],width=0.5,edgecolor='black',lw=1.5)
    for b,v in zip(bars,[pp8,pf8]):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f"{v:.3f}",
                ha='center',va='bottom',fontsize=13,fontweight='bold')
    diff=row["promo_diff"]; sig=row["sig"]
    ax.set_title(f"{cat.capitalize()}\nΔ={diff:+.4f} p={row['p_val']:.3f} {sig}",fontsize=11)
    ax.set_ylabel("Promo Rate (definite exits)")
    ax.set_ylim(0,max(pp8,pf8)*1.4+0.02)
    ax.grid(True,alpha=0.3,axis='y')
    ax.text(0.5,0.92,f"Exits: {row['n_definite_exit']:,} ({row['pct_exit']}%)\nSignal: {row['clearance_signal']}",
            transform=ax.transAxes,ha='center',va='top',fontsize=8,
            bbox=dict(boxstyle='round',facecolor='lightgray',alpha=0.6))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/step06_clearance_v3.png",dpi=150,bbox_inches='tight')
plt.close()
print("\nFigure saved. Summary:")
print(stats[["category","n_definite_exit","promo_diff","p_val","sig","clearance_signal"]].to_string(index=False))
