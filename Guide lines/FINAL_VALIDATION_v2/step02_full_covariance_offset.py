"""
Step 2: Covariance-valid offset CIs using full event-time covariance matrix.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyfixest as pf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
CSV_BASE  = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage5_event_study/rms_fe"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step02_offset_ci"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"

CATS   = ["cereal","snacks","coffee"]
N_RG   = 8
DEPTH  = 0.10
N_DRAWS = 2000
SEED   = 42
REF_TAU = -1
BASELINE_TAUS = list(range(-8,-4))
DURING_TAUS   = list(range(-4,1))
POST_TAUS     = list(range(1,13))

def load_rms(cat, n_rg):
    path = f"{DATA_BASE}/rms_{cat}_analysis.parquet"
    pf_ = pq.ParquetFile(path)
    n = min(n_rg, pf_.num_row_groups)
    cols = ["store_code_uc","upc","week_end","discount_depth","log_units"]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas() for i in range(n)]
    df = pd.concat(frames, ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"])
    df["year_week"] = (df["week_end"].dt.isocalendar().year.astype(str) + "_" +
                       df["week_end"].dt.isocalendar().week.astype(str).str.zfill(2))
    return df

def build_event_study(df, pre=8, post=12, thresh=0.10):
    df = df.sort_values(["store_code_uc","upc","week_end"]).copy()
    g = ["store_code_uc","upc"]
    df["is_promo"] = (df["discount_depth"] >= thresh).astype(int)
    df["next_promo"] = df.groupby(g)["is_promo"].shift(-1, fill_value=0)
    df["is_ep_end"] = (df["is_promo"]==1) & (df["next_promo"]==0)
    
    records = []
    ep_count = 0
    for (store, upc), su_df in df.groupby(g):
        su_df = su_df.reset_index(drop=True)
        ends = su_df[su_df["is_ep_end"]]["week_end"].tolist()
        for end_t in ends:
            su_df["et"] = ((su_df["week_end"] - end_t).dt.days / 7).round().astype(int)
            w = su_df[(su_df["et"] >= -pre) & (su_df["et"] <= post)].copy()
            if len(w) < pre + post:
                continue
            # Require no episode end in pre-window (contamination check)
            pre_ep_ends = w[(w["et"] >= -pre) & (w["et"] < 0) & (w["is_ep_end"])].shape[0]
            if pre_ep_ends > 0:
                continue
            # Require promotion week (tau=0) is actually promoted
            ep_row = w[w["et"]==0]
            if ep_row.empty or ep_row["is_promo"].values[0] == 0:
                continue
            w = w[["store_code_uc","upc","year_week","log_units","et"]].copy()
            w.rename(columns={"et":"event_time"}, inplace=True)
            records.append(w)
            ep_count += 1
    
    if not records:
        return pd.DataFrame(), 0
    out = pd.concat(records, ignore_index=True)
    out = out.drop_duplicates(subset=["store_code_uc","upc","event_time","year_week"])
    return out, ep_count

def run_feols(ep, cat):
    ep = ep.dropna(subset=["log_units","event_time"]).copy()
    ep["event_time"] = ep["event_time"].astype(int)
    ep["su_fe"] = ep["store_code_uc"].astype(str)+"_"+ep["upc"].astype(str)
    ep["sw_fe"] = ep["store_code_uc"].astype(str)+"_"+ep["year_week"].astype(str)
    ep["uw_fe"] = ep["upc"].astype(str)+"_"+ep["year_week"].astype(str)
    fml = f"log_units ~ i(event_time, ref={REF_TAU}) | su_fe + sw_fe + uw_fe"
    try:
        fit = pf.feols(fml, data=ep, vcov={"CRV1":"store_code_uc"})
        return fit
    except Exception as e:
        print(f"  feols error [{cat}]: {e}")
        return None

def compute_ratio_mvn(beta_map, V, tau_order, n_draws, seed, label=""):
    rng = np.random.default_rng(seed)
    b = np.array([beta_map[t] for t in tau_order])
    
    try:
        draws = rng.multivariate_normal(b, V, size=n_draws)
    except Exception:
        draws = rng.multivariate_normal(b, V + 1e-8*np.eye(len(V)), size=n_draws)
    
    idx = {t: i for i, t in enumerate(tau_order)}
    
    ratios = []
    for draw in draws:
        bl_idx = [idx[t] for t in BASELINE_TAUS if t in idx]
        bl_avg = np.mean(draw[bl_idx]) if bl_idx else 0.0
        
        dur_excess = 0.0
        for t in DURING_TAUS:
            if t == REF_TAU:
                dur_excess += (0.0 - bl_avg)
            elif t in idx:
                dur_excess += (draw[idx[t]] - bl_avg)
        
        pst_deficit = sum(draw[idx[t]] - bl_avg for t in POST_TAUS if t in idx)
        
        if dur_excess > 0.001:
            ratios.append(abs(pst_deficit) / dur_excess)
    
    ratios = np.array(ratios)
    return ratios

all_results = []

for cat in CATS:
    print(f"\n{'='*55}")
    print(f"  {cat.upper()}")
    
    ref_df = pd.read_csv(f"{CSV_BASE}/event_study_fe_{cat}.csv", index_col="event_time")
    
    print(f"  Loading parquet (N_RG={N_RG})...")
    df = load_rms(cat, N_RG)
    print(f"  {len(df):,} rows | stores={df['store_code_uc'].nunique()}")
    
    ep, n_ep = build_event_study(df)
    print(f"  Episodes: {n_ep} | obs: {len(ep):,}")
    
    if len(ep) < 200:
        print(f"  SKIP: too few obs")
        continue
    
    fit = run_feols(ep, cat)
    if fit is None:
        continue
    
    tidy  = fit.tidy()
    V_all = fit._vcov
    names = fit._coefnames
    
    # Map event_time -> (index_in_vcov, estimate)
    tau_map = {}
    for i, n in enumerate(names):
        if "event_time::" in n:
            t = int(float(n.split("::")[-1]))
            tau_map[t] = (i, float(tidy.loc[n, "Estimate"]))
    
    tau_order = sorted(tau_map.keys())
    print(f"  Estimated taus: {tau_order}")
    
    if len(tau_order) < 8:
        print(f"  SKIP: too few taus")
        continue
    
    idx_list = [tau_map[t][0] for t in tau_order]
    V_sub    = V_all[np.ix_(idx_list, idx_list)]
    beta_map = {t: tau_map[t][1] for t in tau_order}
    
    # Full covariance MVN
    ratios_full = compute_ratio_mvn(beta_map, V_sub, tau_order, N_DRAWS, SEED)
    
    # Diagonal only MVN (for comparison)
    V_diag = np.diag(np.diag(V_sub))
    ratios_diag = compute_ratio_mvn(beta_map, V_diag, tau_order, N_DRAWS, SEED+1)
    
    # Full-sample ratio from CSV
    bl_ref = [ref_df.loc[t,"coef"] for t in BASELINE_TAUS if t in ref_df.index]
    bl_avg_ref = np.mean(bl_ref)
    
    dur_sum_ref = sum(ref_df.loc[t,"coef"] - bl_avg_ref
                      for t in DURING_TAUS if t in ref_df.index and t != REF_TAU)
    dur_sum_ref += (0.0 - bl_avg_ref)  # add reference tau
    
    pst_sum_ref = sum(ref_df.loc[t,"coef"] - bl_avg_ref
                      for t in POST_TAUS if t in ref_df.index)
    ratio_ref = abs(pst_sum_ref) / dur_sum_ref if dur_sum_ref > 0 else np.nan
    
    pt_full = np.median(ratios_full)
    ci_lo_f = np.percentile(ratios_full, 2.5)
    ci_hi_f = np.percentile(ratios_full, 97.5)
    ci_lo_d = np.percentile(ratios_diag, 2.5)
    ci_hi_d = np.percentile(ratios_diag, 97.5)
    
    print(f"  Full-sample ratio (CSV pts): {ratio_ref:.4f}")
    print(f"  Sandbox full-cov CI: [{ci_lo_f:.4f}, {ci_hi_f:.4f}] (median {pt_full:.4f})")
    print(f"  Sandbox diagonal CI: [{ci_lo_d:.4f}, {ci_hi_d:.4f}]")
    print(f"  During excess: {dur_sum_ref:.4f} | Post deficit: {pst_sum_ref:.4f}")
    
    all_results.append({
        "cat": cat, "n_obs_sandbox": len(ep), "n_episodes": n_ep,
        "n_taus": len(tau_order),
        "during_excess_fullsample": round(dur_sum_ref, 5),
        "post_deficit_fullsample":  round(pst_sum_ref, 5),
        "ratio_fullsample":    round(ratio_ref, 5),
        "ratio_sandbox_median":round(pt_full, 5),
        "ci_lo_full_cov":      round(ci_lo_f, 5),
        "ci_hi_full_cov":      round(ci_hi_f, 5),
        "ci_lo_diagonal":      round(ci_lo_d, 5),
        "ci_hi_diagonal":      round(ci_hi_d, 5),
        "V_diagonal_same_wider": (ci_hi_d - ci_lo_d) >= (ci_hi_f - ci_lo_f),
        "cov_method": "full_CRV1_sandwich",
        "n_draws": N_DRAWS, "N_RG": N_RG,
        "note": "Sandbox (N_RG=8) estimation; full sample needs workstation"
    })
    
    # Figure: event study + ratio distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Step 2: Covariance-Valid Offset CI — {cat.capitalize()}",
                 fontsize=13, fontweight='bold')
    
    # Left: event study plot
    ax = axes[0]
    x  = ref_df.index.tolist()
    y  = ref_df["coef"].values
    lo = ref_df["ci_low"].values
    hi = ref_df["ci_high"].values
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvspan(-8.5, -4.5, alpha=0.12, color='blue', label='Baseline')
    ax.axvspan(-4.5,  0.5, alpha=0.12, color='green',label='During')
    ax.axvspan( 0.5, 12.5, alpha=0.08, color='orange',label='Post')
    ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt='o-', ms=4, lw=1.5,
                color='navy', ecolor='lightblue', elinewidth=1.2,
                capsize=2, label='β_τ ± 95% CI')
    ax.set_xlabel('Event time τ (weeks from promotion end)')
    ax.set_ylabel('β_τ (log units, ref = τ=-1)')
    ax.set_title('Full-sample event study coefficients')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Right: ratio distribution comparison
    ax2 = axes[1]
    bins = np.linspace(0, max(ratios_full.max(), ratios_diag.max())+0.1, 50)
    ax2.hist(ratios_full, bins=bins, alpha=0.6, color='steelblue',
             label=f'Full covariance\nCI=[{ci_lo_f:.3f},{ci_hi_f:.3f}]')
    ax2.hist(ratios_diag, bins=bins, alpha=0.4, color='salmon',
             label=f'Diagonal only\nCI=[{ci_lo_d:.3f},{ci_hi_d:.3f}]')
    ax2.axvline(ratio_ref, color='black', ls='--', lw=2,
                label=f'Full-sample point: {ratio_ref:.3f}')
    ax2.axvline(ci_lo_f, color='steelblue', ls=':', lw=1.5)
    ax2.axvline(ci_hi_f, color='steelblue', ls=':', lw=1.5)
    ax2.set_xlabel('Offset ratio |post| / during')
    ax2.set_ylabel('Count')
    ax2.set_title('MVN draw distribution (2000 draws)\nFull vs Diagonal covariance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/step02_offset_ci_{cat}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved.")

if all_results:
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(f"{OUT_DIR}/offset_ci_full_covariance.csv", index=False)
    print(f"\n{'='*55}")
    print("FINAL RESULTS:")
    print(res_df[["cat","ratio_fullsample","ci_lo_full_cov","ci_hi_full_cov",
                  "ci_lo_diagonal","ci_hi_diagonal","V_diagonal_same_wider"]].to_string(index=False))
else:
    print("No results produced.")
