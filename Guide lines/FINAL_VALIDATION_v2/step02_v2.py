"""Step 2 v2: Full-covariance offset CI (faster implementation)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

DATA_BASE = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage2_variables"
CSV_BASE  = "/sessions/friendly-serene-faraday/mnt/figures and resulrs/06_results/stage5_event_study/rms_fe"
OUT_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/step02_offset_ci"
FIG_DIR   = "/sessions/friendly-serene-faraday/mnt/Guide lines/FINAL_VALIDATION_v2/figures"

CATS   = ["cereal","snacks","coffee"]
N_RG   = 4
DEPTH  = 0.10
N_DRAWS = 2000
SEED   = 42
REF_TAU = -1
BASELINE_TAUS = list(range(-8,-4))
DURING_TAUS   = list(range(-4,1))
POST_TAUS     = list(range(1,13))
ALL_TAUS      = list(range(-8,13))  # -8 to +12

def load_rms_fast(cat, n_rg=4):
    path = f"{DATA_BASE}/rms_{cat}_analysis.parquet"
    pf_ = pq.ParquetFile(path)
    n = min(n_rg, pf_.num_row_groups)
    cols = ["store_code_uc","upc","week_end","discount_depth","log_units"]
    frames = [pf_.read_row_group(i, columns=cols).to_pandas() for i in range(n)]
    df = pd.concat(frames, ignore_index=True)
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df

def build_episode_data_fast(df, thresh=0.10):
    """Vectorized episode detection."""
    df = df.sort_values(["store_code_uc","upc","week_end"]).copy()
    df["is_promo"] = (df["discount_depth"] >= thresh).astype(int)
    
    g = ["store_code_uc","upc"]
    df["next_promo"] = df.groupby(g)["is_promo"].shift(-1, fill_value=0)
    df["is_ep_end"]  = (df["is_promo"]==1) & (df["next_promo"]==0)
    
    # Episode ID within each store-UPC
    df["ep_start"] = ((df["is_promo"]==1) & 
                      (df.groupby(g)["is_promo"].shift(1, fill_value=0)==0)).astype(int)
    df["ep_id"] = df.groupby(g)["ep_start"].cumsum()
    df["ep_id"] = df["ep_id"].where(df["is_promo"]==1, other=np.nan)
    
    # Assign sequential week index within store-UPC
    df["week_idx"] = df.groupby(g).cumcount()
    
    # For episode ends: get their week_idx
    ep_ends = df[df["is_ep_end"]][["store_code_uc","upc","week_idx","ep_id"]].copy()
    ep_ends.columns = ["store_code_uc","upc","end_idx","ep_id"]
    ep_ends["ep_key"] = (ep_ends["store_code_uc"].astype(str) + "_" + 
                         ep_ends["upc"].astype(str) + "_" + 
                         ep_ends["ep_id"].astype(str))
    
    # Cross join within store-UPC to get event times (vectorized)
    df2 = df.merge(ep_ends[["store_code_uc","upc","end_idx","ep_key"]], 
                   on=["store_code_uc","upc"], how="inner")
    df2["event_time"] = df2["week_idx"] - df2["end_idx"]
    df2 = df2[(df2["event_time"] >= -8) & (df2["event_time"] <= 12)].copy()
    df2 = df2.dropna(subset=["log_units"])
    
    print(f"  Episodes: {ep_ends.shape[0]} | obs: {len(df2):,}")
    return df2

def estimate_covariance_ols(ep_data):
    """
    Estimate correlation matrix of event-time dummies via OLS.
    Use simple store-UPC demeaned regression (within estimator).
    """
    ep = ep_data.copy()
    # Within-demean by store-UPC-ep_key
    for col in ["log_units","event_time"]:
        ep[col+"_dm"] = ep[col] - ep.groupby("ep_key")[col].transform("mean")
    
    # Build dummy matrix for event_times != REF_TAU
    taus_to_use = [t for t in ALL_TAUS if t != REF_TAU]
    
    # Create dummy variables
    X_rows = []
    for t in taus_to_use:
        X_rows.append((ep["event_time"] == t).astype(float).values)
    X = np.array(X_rows).T  # (n_obs, n_taus)
    y = ep["log_units_dm"].values
    
    # OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    
    try:
        beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    except Exception:
        return None, None, None
    
    # Residuals
    resid = y - X @ beta
    n, k  = len(y), len(taus_to_use)
    s2    = np.sum(resid**2) / (n - k)
    
    # Homoskedastic covariance (for correlation structure only)
    try:
        V_ols = s2 * np.linalg.inv(XtX)
    except Exception:
        V_ols = s2 * np.linalg.pinv(XtX)
    
    # Extract correlation matrix
    se_ols = np.sqrt(np.diag(V_ols))
    corr = V_ols / np.outer(se_ols, se_ols)
    
    return beta, corr, taus_to_use

def mvn_ratio(beta_hat, V_full, tau_order, n_draws=2000, seed=42):
    """Compute 2000 MVN draws and compute ratio for each."""
    rng = np.random.default_rng(seed)
    
    # Include REF_TAU=0 in tau_order (reference constraint)
    # tau_order does NOT include REF_TAU — we add it as zero
    full_tau_order = tau_order  # REF_TAU not in tau_order
    idx = {t: i for i, t in enumerate(full_tau_order)}
    
    try:
        draws = rng.multivariate_normal(beta_hat, V_full, size=n_draws)
    except np.linalg.LinAlgError:
        V_full = V_full + 1e-8 * np.eye(len(V_full))
        draws = rng.multivariate_normal(beta_hat, V_full, size=n_draws)
    
    ratios = []
    for d in draws:
        bl_idx = [idx[t] for t in BASELINE_TAUS if t in idx]
        bl_avg = np.mean(d[bl_idx]) if bl_idx else 0.0
        
        # During: sum(beta_t - bl_avg) for t in DURING_TAUS
        # REF_TAU is 0 by constraint
        dur = sum(d[idx[t]] - bl_avg for t in DURING_TAUS if t in idx and t != REF_TAU)
        dur += (0.0 - bl_avg)  # REF_TAU contributes (0 - bl_avg)
        
        pst = sum(d[idx[t]] - bl_avg for t in POST_TAUS if t in idx)
        
        if dur > 1e-4:
            ratios.append(abs(pst) / dur)
    
    ratios = np.array(ratios)
    return ratios

all_results = []

for cat in CATS:
    print(f"\n{'='*55}\n  {cat.upper()}")
    
    # Full-sample event study coefficients (from prior full run)
    ref_df = pd.read_csv(f"{CSV_BASE}/event_study_fe_{cat}.csv", index_col="event_time")
    
    print(f"  Loading parquet (N_RG={N_RG})...")
    df = load_rms_fast(cat, N_RG)
    print(f"  {len(df):,} rows")
    
    ep_data = build_episode_data_fast(df)
    
    if len(ep_data) < 100:
        print("  SKIP: too few obs")
        continue
    
    print("  Estimating OLS covariance structure...")
    beta_ols, corr, taus_used = estimate_covariance_ols(ep_data)
    
    if corr is None:
        print("  SKIP: covariance estimation failed")
        continue
    
    print(f"  Taus used: {taus_used}")
    
    # ── Hybrid approach: use full-sample SEs from CSV + OLS correlation ──
    # This is the key methodological contribution:
    # We use the properly estimated correlation structure to build V
    
    tau_order = taus_used  # does not include REF_TAU=-1
    
    # Get full-sample point estimates for each tau
    beta_full = np.array([
        ref_df.loc[t, "coef"] if t in ref_df.index else 0.0
        for t in tau_order
    ])
    se_full = np.array([
        ref_df.loc[t, "se"] if t in ref_df.index else 1.0
        for t in tau_order
    ])
    
    # Hybrid full covariance: V = diag(se_full) @ corr @ diag(se_full)
    V_hybrid = np.outer(se_full, se_full) * corr
    
    # Diagonal-only covariance
    V_diag = np.diag(se_full**2)
    
    # MVN draws
    ratios_hybrid = mvn_ratio(beta_full, V_hybrid, tau_order, N_DRAWS, SEED)
    ratios_diag   = mvn_ratio(beta_full, V_diag,   tau_order, N_DRAWS, SEED+1)
    
    # Point estimates from full-sample CSV
    bl_ref = np.mean([ref_df.loc[t,"coef"] for t in BASELINE_TAUS if t in ref_df.index])
    dur_ref = sum(ref_df.loc[t,"coef"] - bl_ref for t in DURING_TAUS 
                  if t in ref_df.index and t != REF_TAU) + (0.0 - bl_ref)
    pst_ref = sum(ref_df.loc[t,"coef"] - bl_ref for t in POST_TAUS if t in ref_df.index)
    ratio_ref = abs(pst_ref) / dur_ref if dur_ref > 0 else np.nan
    
    # CIs
    pt_h  = np.median(ratios_hybrid)
    lo_h  = np.percentile(ratios_hybrid, 2.5)
    hi_h  = np.percentile(ratios_hybrid, 97.5)
    lo_d  = np.percentile(ratios_diag, 2.5)
    hi_d  = np.percentile(ratios_diag, 97.5)
    
    print(f"  Full-sample ratio: {ratio_ref:.4f}")
    print(f"  Full-cov CI (hybrid): [{lo_h:.4f}, {hi_h:.4f}]")
    print(f"  Diagonal CI:          [{lo_d:.4f}, {hi_d:.4f}]")
    print(f"  Diagonal wider: {(hi_d-lo_d) >= (hi_h-lo_h)}")
    
    all_results.append({
        "cat": cat,
        "n_obs_sandbox": len(ep_data),
        "baseline_window": "tau in {-8,-7,-6,-5}",
        "during_window":   "tau in {-4,-3,-2,-1,0}",
        "post_window":     "tau in {+1,...,+12}",
        "during_excess":   round(dur_ref, 5),
        "post_deficit":    round(pst_ref, 5),
        "ratio_point":     round(ratio_ref, 5),
        "ci_lo_full_cov":  round(lo_h, 5),
        "ci_hi_full_cov":  round(hi_h, 5),
        "ci_lo_diagonal":  round(lo_d, 5),
        "ci_hi_diagonal":  round(hi_d, 5),
        "diagonal_wider": (hi_d-lo_d) >= (hi_h-lo_h),
        "method": "hybrid: full-sample beta/SE + OLS correlation structure",
        "n_draws": N_DRAWS,
        "interpretation": (
            "RATIO < 1: post-promo deficit smaller than during-promo excess "
            "(incomplete offset = genuine incremental demand)"
        )
    })
    
    # ── Figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Step 2 — Covariance-Valid Offset CI: {cat.capitalize()}",
                 fontsize=13, fontweight='bold')
    
    # 1. Event study plot
    ax = axes[0]
    x = ref_df.index.tolist()
    y = ref_df["coef"].values
    lo_ev = ref_df["ci_low"].values
    hi_ev = ref_df["ci_high"].values
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvspan(-8.5,-4.5, alpha=0.12, color='royalblue', label='Baseline')
    ax.axvspan(-4.5, 0.5, alpha=0.12, color='green',     label='During')
    ax.axvspan( 0.5,12.5, alpha=0.08, color='orange',    label='Post')
    ax.errorbar(x, y, yerr=[y-lo_ev, hi_ev-y], fmt='o-', ms=4, lw=1.5,
                color='navy', ecolor='lightsteelblue', elinewidth=1.2,
                capsize=2, label='β_τ ± 95% CI')
    ax.set_xlabel('Event time τ (weeks from promo end)')
    ax.set_ylabel('β_τ (log units, ref = τ=-1)')
    ax.set_title(f'Full-sample event study\nduring_excess={dur_ref:.3f}, post_deficit={pst_ref:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Correlation heatmap
    ax2 = axes[1]
    im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    # Label every 4th tick
    step = max(1, len(tau_order)//6)
    ticks = list(range(0, len(tau_order), step))
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels([str(tau_order[i]) for i in ticks], fontsize=7)
    ax2.set_yticklabels([str(tau_order[i]) for i in ticks], fontsize=7)
    ax2.set_title('OLS correlation structure\nof event-time dummies')
    ax2.set_xlabel('Event time τ')
    ax2.set_ylabel('Event time τ')
    
    # 3. Ratio distribution
    ax3 = axes[2]
    xmax = max(ratios_hybrid.max(), ratios_diag.max(), ratio_ref) + 0.05
    bins = np.linspace(0, min(xmax, 3.0), 60)
    ax3.hist(ratios_hybrid, bins=bins, alpha=0.6, color='steelblue',
             label=f'Full cov (hybrid)\n95% CI=[{lo_h:.3f},{hi_h:.3f}]')
    ax3.hist(ratios_diag,   bins=bins, alpha=0.4, color='salmon',
             label=f'Diagonal only\n95% CI=[{lo_d:.3f},{hi_d:.3f}]')
    ax3.axvline(ratio_ref, color='black', ls='--', lw=2,
                label=f'Point est: {ratio_ref:.3f}')
    ax3.axvline(lo_h, color='steelblue', ls=':', lw=1.5)
    ax3.axvline(hi_h, color='steelblue', ls=':', lw=1.5)
    ax3.axvline(1.0, color='red', ls='-.', lw=1.5, alpha=0.7,
                label='Ratio=1 (full offset)')
    ax3.set_xlabel('Offset ratio |post deficit| / during excess')
    ax3.set_ylabel('Count (of 2000 MVN draws)')
    ax3.set_title(f'Ratio distribution\nFull cov vs Diagonal')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/step02_offset_ci_{cat}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved.")

if all_results:
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(f"{OUT_DIR}/offset_ci_full_covariance.csv", index=False)
    print(f"\n{'='*55}")
    print("RESULTS SAVED:")
    print(res_df[["cat","ratio_point","ci_lo_full_cov","ci_hi_full_cov",
                  "ci_lo_diagonal","ci_hi_diagonal","diagonal_wider"]].to_string(index=False))
else:
    print("No results.")
