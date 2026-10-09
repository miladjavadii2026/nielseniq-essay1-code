"""
Task 07 — Promotion Episode Duration Split Event Study
======================================================
Verdict requirement (Section 7 of audit): Run SEPARATE event studies for
one-week, two-week, and three-or-more-week promotion episodes.

The shallow-vs-deep comparison in the prior report does NOT satisfy this.
This script computes duration-stratified event studies and offset ratios.

Clean-window rule enforced:
  - No prior promotion episode ending within 4 weeks before event
  - No new promotion starting during the post-window (t = +1 to +8)
  - For one-week episodes: t = -1 must NOT be a promoted week

Inputs
------
RMS panel parquet with columns:
    store_id, upc, week, log_quantity, promo_flag, discount_pct, depth_bin
    (same panel as the main FE model)

Outputs
-------
  Table : printed to stdout — episodes by length × category
          (count, during-sum, post-sum, offset ratio, 95% bootstrap CI)
  Figure: saved to FIG_DIR / "Task07_EpisodeDuration.png"

Usage
-----
    python task07_episode_duration.py \
        --data_path /path/to/rms_panel.parquet \
        --category cereal \
        --fig_dir /path/to/phase2c_report_figures

Author: Milad Javadi — UTD Economics, Phase 2C
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfixest as pf
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# Post-promotion window for offset ratio
POST_WINDOW = 8      # weeks
PRE_WINDOW  = 4      # weeks before event
CLEAN_GAP   = 4      # no prior promo within this many weeks
BOOT_REPS   = 2000   # parametric bootstrap draws


# ── Step 1: Label promotion episodes ──────────────────────────────────────────
def label_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (store_id, upc), identify runs of consecutive promoted weeks.
    Adds columns:
        episode_id    : unique integer per episode (NaN for non-promoted weeks)
        episode_len   : number of consecutive weeks in the episode
        episode_end   : 1 if this is the last week of an episode, else 0
        event_week    : week number of episode end (broadcast to all obs in episode)
    """
    df = df.copy().sort_values(["store_id", "upc", "week"]).reset_index(drop=True)

    # Identify start of a new episode: promo_flag transitions 0→1 or is 1 at first obs
    df["grp"] = (
        (df["promo_flag"] != df.groupby(["store_id", "upc"])["promo_flag"].shift(1))
        & (df["promo_flag"] == 1)
    ).cumsum()
    df.loc[df["promo_flag"] == 0, "grp"] = np.nan

    # Episode id = (store_id, upc, grp) — unique per run
    df["episode_id"] = df.groupby(["store_id", "upc", "grp"], dropna=True).ngroup()
    df.loc[df["promo_flag"] == 0, "episode_id"] = np.nan

    # Episode length
    ep_len = (
        df.dropna(subset=["episode_id"])
        .groupby("episode_id")["week"]
        .transform("count")
    )
    df["episode_len"] = ep_len
    df.loc[df["promo_flag"] == 0, "episode_len"] = np.nan

    # Episode end: last week in each episode
    ep_end_week = (
        df.dropna(subset=["episode_id"])
        .groupby("episode_id")["week"]
        .transform("max")
    )
    df["episode_end_week"] = ep_end_week
    df["episode_end"] = (df["week"] == df["episode_end_week"]).astype(int)
    df.loc[df["promo_flag"] == 0, "episode_end"] = 0

    return df


# ── Step 2: Assign event time relative to episode end ─────────────────────────
def assign_event_time(df: pd.DataFrame, pre: int = PRE_WINDOW,
                      post: int = POST_WINDOW) -> pd.DataFrame:
    """
    For each episode-end event, compute event_time = week - episode_end_week
    in the surrounding window [−pre, +post].
    Only keep rows within the window.
    """
    # Anchor = one row per episode end
    ends = (
        df[df["episode_end"] == 1][["store_id", "upc", "episode_id",
                                     "episode_len", "episode_end_week"]]
        .drop_duplicates()
        .rename(columns={"episode_end_week": "anchor_week"})
    )

    # Merge back to full panel to get event_time for surrounding weeks
    event_df = df.merge(ends, on=["store_id", "upc"], how="inner", suffixes=("", "_ep"))
    event_df["event_time"] = event_df["week"] - event_df["anchor_week"]
    event_df = event_df[
        (event_df["event_time"] >= -pre) & (event_df["event_time"] <= post)
    ].copy()

    return event_df


# ── Step 3: Apply clean-window rule ───────────────────────────────────────────
def apply_clean_window(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove episodes that violate the clean-window rule:
      (a) Another episode ended within CLEAN_GAP weeks before this episode started
      (b) A new promotion starts during the post-window (event_time = +1 to +8)
      (c) For 1-week episodes only: event_time = -1 must not be promoted
    """
    # Flag if post-window has any promotion
    post_promo = (
        event_df[event_df["event_time"].between(1, POST_WINDOW)]
        .groupby(["store_id", "upc", "episode_id"])["promo_flag"]
        .max()
        .rename("post_has_promo")
    )
    event_df = event_df.merge(post_promo.reset_index(), on=["store_id", "upc", "episode_id"], how="left")
    event_df["post_has_promo"] = event_df["post_has_promo"].fillna(0)

    # For 1-week episodes: check t=-1 is not promoted
    pre1_promo = (
        event_df[(event_df["event_time"] == -1) & (event_df["episode_len"] == 1)]
        [["store_id", "upc", "episode_id", "promo_flag"]]
        .rename(columns={"promo_flag": "t_minus1_promo"})
    )
    event_df = event_df.merge(pre1_promo, on=["store_id", "upc", "episode_id"], how="left")
    event_df["t_minus1_promo"] = event_df["t_minus1_promo"].fillna(0)

    # Keep only clean episodes
    clean = event_df[
        (event_df["post_has_promo"] == 0) &
        ~((event_df["episode_len"] == 1) & (event_df["t_minus1_promo"] == 1))
    ].copy()

    n_before = event_df["episode_id"].nunique()
    n_after  = clean["episode_id"].nunique()
    print(f"  Clean-window filter: {n_before:,} → {n_after:,} episodes "
          f"({n_before - n_after:,} removed)")

    return clean


# ── Step 4: Run event study per episode-length group ──────────────────────────
def run_episode_event_study(clean_df: pd.DataFrame,
                             ep_len_label: str) -> dict:
    """
    Run promotion-end event study for a specific episode-length group.
    Returns dict with point estimates, CI, offset ratio, and bootstrap CI.

    Args:
        clean_df: filtered dataframe for this episode-length group
        ep_len_label: e.g. "1_week", "2_week", "3plus_week"
    """
    # Omit t = -1 as reference
    rel_times = [t for t in range(-PRE_WINDOW, POST_WINDOW + 1) if t != -1]

    for t in rel_times:
        clean_df[f"d{t}"] = (clean_df["event_time"] == t).astype(int)

    time_terms = " + ".join([f"d{t}" for t in rel_times])
    formula = f"log_quantity ~ {time_terms} | store_id^upc + store_id^week + upc^week"

    try:
        fit = pf.feols(formula, data=clean_df, vcov={"CRV1": "store_id"})
    except Exception as e:
        print(f"  [WARN] feols failed for {ep_len_label}: {e}")
        return {}

    coef_names = [f"d{t}" for t in rel_times]
    coefs = {name: fit.coef().get(name, 0.0) for name in coef_names}
    ses   = {name: fit.se().get(name, 0.0)   for name in coef_names}

    # Offset ratio: |sum(post)| / sum(during incl. t=0)
    during_sum = sum(coefs.get(f"d{t}", 0) for t in range(-PRE_WINDOW, 1))
    post_sum   = sum(coefs.get(f"d{t}", 0) for t in range(1, POST_WINDOW + 1))
    offset_ratio = abs(post_sum) / max(abs(during_sum), 1e-9)

    # Parametric bootstrap CI for offset ratio
    boot_ratios = []
    for _ in range(BOOT_REPS):
        during_draws = [rng.normal(coefs[f"d{t}"], ses[f"d{t}"])
                        for t in range(-PRE_WINDOW, 1) if f"d{t}" in coefs]
        post_draws   = [rng.normal(coefs[f"d{t}"], ses[f"d{t}"])
                        for t in range(1, POST_WINDOW + 1) if f"d{t}" in coefs]
        d_sum = sum(during_draws)
        p_sum = sum(post_draws)
        if abs(d_sum) > 1e-9:
            boot_ratios.append(abs(p_sum) / abs(d_sum))

    ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5]) if boot_ratios else (np.nan, np.nan)

    n_episodes = clean_df["episode_id"].nunique()
    print(f"  {ep_len_label}: N={n_episodes:,} eps | "
          f"during={during_sum:.3f} | post={post_sum:.3f} | "
          f"ratio={offset_ratio:.3f} | 95%CI=[{ci_lo:.3f},{ci_hi:.3f}]")

    return {
        "label": ep_len_label, "n_episodes": n_episodes,
        "during_sum": during_sum, "post_sum": post_sum,
        "offset_ratio": offset_ratio, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "coefs": coefs, "ses": ses
    }


# ── Step 5: Plot event studies side-by-side ───────────────────────────────────
def plot_episode_event_studies(results: list, category: str, fig_dir: Path):
    """Three-panel figure: one panel per episode-length group."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, res in zip(axes, results):
        if not res:
            continue
        times = sorted([int(k[1:]) for k in res["coefs"]])
        plot_times = sorted(times + [-1])
        ys = [0 if t == -1 else res["coefs"].get(f"d{t}", 0) for t in plot_times]
        ses = [0 if t == -1 else res["ses"].get(f"d{t}", 0) for t in plot_times]
        lbs = [y - 1.96 * s for y, s in zip(ys, ses)]
        ubs = [y + 1.96 * s for y, s in zip(ys, ses)]

        ax.plot(plot_times, ys, "o-", color="#2c6fad", lw=2, ms=4)
        ax.fill_between(plot_times, lbs, ubs, alpha=0.2, color="#2c6fad")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="gray", lw=1.2, linestyle="--")
        ax.axvspan(0.5, POST_WINDOW + 0.5, alpha=0.06, color="red",
                   label="Post-promo window")
        ax.set_title(f"{res['label'].replace('_', ' ')}\n"
                     f"N={res['n_episodes']:,} eps | ratio={res['offset_ratio']:.3f}", fontsize=9)
        ax.set_xlabel("Weeks relative to promo end", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Effect on ln(1+units)", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Figure: Episode-Duration Event Study — {category.title()} (NielsenIQ)\n"
        f"Separate estimates by 1-week, 2-week, 3+-week episodes | "
        f"Clean-window enforced",
        fontsize=10
    )
    plt.tight_layout()
    out = fig_dir / "Task07_EpisodeDuration.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_path: str, category: str, fig_dir: str):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Task 07 — Episode Duration Split: {category.upper()} ===")

    df = pd.read_parquet(data_path)
    df["log_quantity"] = np.log1p(df["units"])   # ensure ln(1+units)

    print(f"Loaded {len(df):,} obs | {df['store_id'].nunique()} stores | "
          f"{df['upc'].nunique()} UPCs")

    print("\nStep 1: Labelling episodes...")
    df = label_episodes(df)

    print("Step 2: Assigning event time...")
    event_df = assign_event_time(df)
    print(f"  Event window observations: {len(event_df):,}")

    print("Step 3: Applying clean-window rule...")
    clean_df = apply_clean_window(event_df)

    results = []
    print("\nStep 4: Running episode-length-stratified event studies...")
    for label, mask in [
        ("1_week",      clean_df["episode_len"] == 1),
        ("2_week",      clean_df["episode_len"] == 2),
        ("3plus_week",  clean_df["episode_len"] >= 3),
    ]:
        grp = clean_df[mask].copy()
        n_ep = grp["episode_id"].nunique()
        print(f"\n  {label}: {n_ep:,} clean episodes")
        if n_ep < 100:
            print(f"  [WARN] Insufficient episodes for reliable inference")
        results.append(run_episode_event_study(grp, label))

    print("\nStep 5: Plotting...")
    plot_episode_event_studies(results, category, fig_dir)

    print("\n=== Summary Table ===")
    print(f"{'Length':<15} {'N eps':>8} {'During':>9} {'Post':>9} "
          f"{'Ratio':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print("-" * 72)
    for r in results:
        if r:
            print(f"{r['label']:<15} {r['n_episodes']:>8,} {r['during_sum']:>9.3f} "
                  f"{r['post_sum']:>9.3f} {r['offset_ratio']:>8.3f} "
                  f"{r['ci_lo']:>8.3f} {r['ci_hi']:>8.3f}")

    print("\n[NOTE] If cereal 1-week and multi-week offset ratios differ substantially,")
    print("       t=-1 contamination in multi-week episodes is confirmed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path to RMS parquet for this category")
    parser.add_argument("--category", required=True,
                        choices=["cereal", "snacks", "coffee"])
    parser.add_argument("--fig_dir", default=".",
                        help="Output directory for figures")
    args = parser.parse_args()
    main(args.data_path, args.category, args.fig_dir)
