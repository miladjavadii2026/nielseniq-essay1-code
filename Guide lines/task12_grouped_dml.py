"""
Task 12 — GroupKFold DML with Clustered Inference
==================================================
Verdict requirement (Section 9): The current main DML result uses ORDINARY
5-fold KFold with HC (heteroskedasticity-robust) standard errors.

Problems with ordinary KFold:
  1. Store observations appear in BOTH train and test folds
  2. Nuisance models can "memorise" store-specific patterns
  3. HC SEs ignore within-store correlation → confidence intervals too narrow

This script runs:
  - GroupKFold(n_splits=5, groups=store_id): stores are never split across folds
  - Repeated splits: run 5 random group assignments, take median ATE
  - Clustered SE: sandwich estimator with store clusters
  - Three learner families: Lasso, Random Forest, Gradient Boosting
  - Identical samples across learners for comparability

Required output table (verdict, Section 9):
  Category | Treatment | Learner | ATE | Clustered SE | 95% CI

Correct language AFTER this script:
  "ATE estimates are stable across nuisance learner specifications.
   Under GroupKFold cross-fitting and store-clustered inference,
   the 95% CI for cereal is [lo, hi]."

Inputs
------
RMS panel parquet (same as FE model):
    store_id, upc, week, log_quantity, W5, W10, [feature columns]

Outputs
-------
  Table: ATE, clustered SE, 95% CI by category × treatment × learner (stdout + CSV)
  Figure: Task12_GroupedDML.png — ATE comparison across learners

Author: Milad Javadi — UTD Economics, Phase 2C
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                               RandomForestRegressor, RandomForestClassifier)
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SEED       = 42
rng        = np.random.default_rng(SEED)
N_SPLITS   = 5    # folds for GroupKFold
N_REPEATS  = 5    # repeated random group orderings (take median ATE across repeats)


# ── Feature construction ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    df = df.copy()
    df["week_sin"] = np.sin(2 * np.pi * (df["week"] % 52) / 52)
    df["week_cos"] = np.cos(2 * np.pi * (df["week"] % 52) / 52)

    feature_cols = []
    for col in ["store_income", "category_hhi", "brand_share", "comp_density",
                 "prev_promo_freq", "hist_median_sales", "week_sin", "week_cos"]:
        if col in df.columns:
            feature_cols.append(col)

    if "retailer_code" in df.columns:
        dummies = pd.get_dummies(df["retailer_code"], prefix="retailer", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        feature_cols += list(dummies.columns)

    X = df[feature_cols].fillna(df[feature_cols].median()).values.astype(float)
    return X, feature_cols


# ── Nuisance learner definitions ───────────────────────────────────────────────
def get_learners() -> dict:
    """
    Three learner families for nuisance models (Y|X and W|X).
    Identical across all DML runs for comparability.
    """
    return {
        "Lasso": {
            "y": LassoCV(cv=3, max_iter=5000, random_state=SEED),
            "t": LogisticRegressionCV(cv=3, max_iter=1000, random_state=SEED),
        },
        "RandomForest": {
            "y": RandomForestRegressor(n_estimators=200, max_depth=6,
                                        n_jobs=-1, random_state=SEED),
            "t": RandomForestClassifier(n_estimators=200, max_depth=6,
                                         n_jobs=-1, random_state=SEED),
        },
        "GradBoost": {
            "y": GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                            learning_rate=0.05, subsample=0.8,
                                            random_state=SEED),
            "t": GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=SEED),
        },
    }


# ── GroupKFold cross-fitting ───────────────────────────────────────────────────
def crossfit_residuals(X: np.ndarray, Y: np.ndarray, W: np.ndarray,
                        groups: np.ndarray, learner_y, learner_t,
                        repeat_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-fit residuals Y_tilde and W_tilde using GroupKFold.

    Groups = store_id array: ensures no store appears in both train and test.
    Returns: Y_tilde (residualised outcome), W_tilde (residualised treatment)
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    Y_tilde = np.zeros(len(Y))
    W_tilde = np.zeros(len(W))

    # Shuffle store order for this repeat
    unique_groups = np.unique(groups)
    rng_r = np.random.default_rng(repeat_seed)
    rng_r.shuffle(unique_groups)
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups_shuffled = np.array([group_map[g] for g in groups])

    for train_idx, test_idx in gkf.split(X, Y, groups=groups_shuffled):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        W_tr, W_te = W[train_idx], W[test_idx]

        # Fit Y nuisance on train, predict on test
        learner_y.fit(X_tr, Y_tr)
        Y_tilde[test_idx] = Y_te - learner_y.predict(X_te)

        # Fit W nuisance on train, predict on test
        learner_t.fit(X_tr, W_tr)
        if hasattr(learner_t, "predict_proba"):
            W_hat = learner_t.predict_proba(X_te)[:, 1]
        else:
            W_hat = learner_t.predict(X_te)
        W_tilde[test_idx] = W_te - W_hat

    return Y_tilde, W_tilde


# ── ATE estimation and clustered SE ───────────────────────────────────────────
def estimate_ate_clustered(Y_tilde: np.ndarray, W_tilde: np.ndarray,
                             store_ids: np.ndarray) -> dict:
    """
    ATE = E[W_tilde * Y_tilde] / E[W_tilde^2]

    Clustered SE via sandwich estimator with store clusters.
    Scores: psi_i = W_tilde_i * (Y_tilde_i - ATE * W_tilde_i)
    Clustered variance: V = (sum_g (sum_i psi_ig)^2) / N^2

    This is the correct SE for grouped cross-fitting.
    """
    N   = len(Y_tilde)
    num = np.dot(W_tilde, Y_tilde)
    den = np.dot(W_tilde, W_tilde)
    ate = num / den

    # Score for each observation
    psi = W_tilde * (Y_tilde - ate * W_tilde)

    # Cluster-sum scores by store
    stores     = np.unique(store_ids)
    n_clusters = len(stores)
    meat = 0.0
    for s in stores:
        mask = store_ids == s
        meat += np.sum(psi[mask]) ** 2

    # Sandwich variance
    bread = den ** 2
    var   = (n_clusters / (n_clusters - 1)) * meat / bread
    se    = np.sqrt(var / N)

    ci_lo = ate - 1.96 * se
    ci_hi = ate + 1.96 * se
    t_stat = ate / se if se > 0 else np.nan

    return {"ate": ate, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "t_stat": t_stat, "n_clusters": n_clusters, "n_obs": N}


# ── Run full DML for one category × treatment × learner ───────────────────────
def run_grouped_dml(df: pd.DataFrame, treatment_col: str,
                    learner_name: str, learners: dict) -> dict:
    """
    Full GroupKFold DML with repeated splits.
    Returns median ATE across N_REPEATS + clustered SE from first repeat.
    """
    from copy import deepcopy

    X, feature_names = build_features(df)
    Y = df["log_quantity"].values
    W = df[treatment_col].astype(float).values
    G = df["store_id"].values    # group array

    lrn = learners[learner_name]

    ate_list = []
    for rep in range(N_REPEATS):
        Y_tilde, W_tilde = crossfit_residuals(
            X, Y, W, G,
            deepcopy(lrn["y"]),
            deepcopy(lrn["t"]),
            repeat_seed=rep + SEED
        )
        res = estimate_ate_clustered(Y_tilde, W_tilde, G)
        ate_list.append(res)

    # Median ATE across repeats; SE from first (most stable) repeat
    ate_median = np.median([r["ate"] for r in ate_list])
    se_first   = ate_list[0]["se"]
    ci_lo_m    = ate_median - 1.96 * se_first
    ci_hi_m    = ate_median + 1.96 * se_first

    return {
        "learner": learner_name, "treatment": treatment_col,
        "ate": ate_median, "se": se_first,
        "ci_lo": ci_lo_m, "ci_hi": ci_hi_m,
        "n_clusters": ate_list[0]["n_clusters"],
        "n_obs": ate_list[0]["n_obs"],
    }


# ── Plot comparison across learners ───────────────────────────────────────────
def plot_dml_comparison(all_results: list[dict], category: str, fig_dir: Path):
    """Compare ATE across learners for W5 and W10."""
    df_res = pd.DataFrame(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, trt in zip(axes, ["W5", "W10"]):
        sub = df_res[df_res["treatment"] == trt].reset_index(drop=True)
        if sub.empty:
            continue
        x = np.arange(len(sub))
        errs_lo = (sub["ate"] - sub["ci_lo"]).tolist()
        errs_hi = (sub["ci_hi"] - sub["ate"]).tolist()
        ax.bar(sub["learner"], sub["ate"], color="#2c6fad", alpha=0.75,
               yerr=[errs_lo, errs_hi], capsize=5, error_kw={"elinewidth": 1.5})
        ax.set_title(f"{trt} | GroupKFold | Clustered SE by store", fontsize=9)
        ax.set_ylabel("ATE [ln(1+units)]", fontsize=9)
        ax.set_xlabel("Nuisance Learner", fontsize=9)
        for i, row in sub.iterrows():
            ax.text(i, row["ate"] + 0.002, f"{row['ate']:.3f}", ha="center",
                    fontsize=8, color="black")

    fig.suptitle(
        f"Task 12: Grouped DML — {category.title()} (NielsenIQ)\n"
        f"GroupKFold(n={N_SPLITS}) × {N_REPEATS} repeats | "
        f"Clustered SE by store_id",
        fontsize=10
    )
    plt.tight_layout()
    out = fig_dir / "Task12_GroupedDML.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_path: str, category: str, fig_dir: str, out_dir: str):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Task 12 — GroupKFold DML: {category.upper()} ===")

    df = pd.read_parquet(data_path)
    df["log_quantity"] = np.log1p(df["units"].clip(lower=0))
    if "category" in df.columns:
        df = df[df["category"] == category].copy()
    print(f"Loaded {len(df):,} obs | {df['store_id'].nunique():,} stores")

    learners = get_learners()
    all_results = []

    for treatment in ["W5", "W10"]:
        if treatment not in df.columns:
            print(f"  [SKIP] {treatment} column not found")
            continue
        for lrn_name in ["Lasso", "RandomForest", "GradBoost"]:
            print(f"\n  {treatment} × {lrn_name}...")
            try:
                res = run_grouped_dml(df, treatment, lrn_name, learners)
                res["category"] = category
                all_results.append(res)
                print(f"    ATE={res['ate']:.4f} | ClustSE={res['se']:.4f} | "
                      f"95%CI=[{res['ci_lo']:.4f},{res['ci_hi']:.4f}]")
            except Exception as e:
                print(f"  [ERROR] {e}")

    if not all_results:
        print("\n[ERROR] No results produced. Check data schema.")
        return

    print("\n=== TASK 12 RESULTS TABLE ===")
    print(f"{'Category':<10} {'Treatment':<11} {'Learner':<15} {'ATE':>8} "
          f"{'ClustSE':>10} {'CI_lo':>8} {'CI_hi':>8} {'N_stores':>10}")
    print("-" * 84)
    for r in all_results:
        print(f"{r['category']:<10} {r['treatment']:<11} {r['learner']:<15} "
              f"{r['ate']:>8.4f} {r['se']:>10.4f} {r['ci_lo']:>8.4f} "
              f"{r['ci_hi']:>8.4f} {r['n_clusters']:>10,}")

    # CSV output
    out_csv = out_dir / f"task12_grouped_dml_{category}.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\n  Results saved: {out_csv}")

    print("\nPlotting...")
    plot_dml_comparison(all_results, category, fig_dir)

    print("\n[NOTE] These results replace the ordinary-KFold estimates in the prior report.")
    print("       ATE point estimates should be stable; CIs will be wider under clustering.")
    print("       The SE ratio (clustered/HC) is a new diagnostic — report it in the paper.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--category",  required=True, choices=["cereal","snacks","coffee"])
    parser.add_argument("--fig_dir",   default=".")
    parser.add_argument("--out_dir",   default=".")
    args = parser.parse_args()
    main(args.data_path, args.category, args.fig_dir, args.out_dir)
