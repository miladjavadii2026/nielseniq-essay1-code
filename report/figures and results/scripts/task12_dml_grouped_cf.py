"""
task12_dml_grouped_cf.py
=========================
PURPOSE
-------
Confirm that the existing DML estimates use:
  1. Grouped (clustered) cross-fitting: fold assignment respects store-level
     clustering so that all observations from one store appear in the same fold
  2. Clustered inference: standard errors clustered at store level (same unit
     as the promotion allocation decision)
  3. Sensitivity to fold count: compare K=5 (default) vs K=10 vs K=3

If the existing stage9 DML was run WITHOUT grouped cross-fitting, Neyman
orthogonality holds but the variance estimator is inconsistent (observations
from the same store appear in both training and test folds, which inflates
precision). This script verifies the implementation and re-runs with correct
grouped cross-fitting if needed.

ECONOMIC MOTIVATION
--------------------
The DML identification assumption is that Y(1) - Y(0) ⊥ W | X.
With clustered cross-fitting: fold i gets all stores not in group i,
so the nuisance estimator (Yhat, What) is truly out-of-sample for each store.
Without it: same-store autocorrelation can leak signal into the nuisance
estimator, biasing the residual. This is especially important here because
stores within a DMA share promotion schedules (the manufacturer promotion
timing affects all stores in the market).

INPUTS
------
  - stage9_dml_robustness/tables/dml_robustness_{cat}.csv
  - stage9_dml_robustness/tables/dml_sensitivity_{cat}.csv
  (Already produced by stage9_dml_robustness.py)

  OR re-runs DML from scratch if grouped cross-fitting is not confirmed.

OUTPUTS
-------
  - task12_dml_verification.csv     (verification results per category)
  - task12_dml_fold_sensitivity.csv (K=3,5,10 comparison)
  - task12_dml_diagnostic.txt       (human-readable audit narrative)
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────
STAGE2  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
STAGE9  = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage9_dml_robustness"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task12_dml_verification"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES  = ["cereal", "coffee", "snacks"]
SAMPLE_FRAC = 0.03
FOLD_COUNTS = [3, 5, 10]
RANDOM_SEED = 42

print("=" * 65)
print("Task 12: DML Grouped Cross-Fitting and Clustered Inference Audit")
print("=" * 65)

# ── Step 1: Audit existing stage9 outputs ────────────────────────────────

def load_stage9(cat: str) -> dict:
    """Try to load existing DML robustness results."""
    result = {"found": False, "paths_checked": []}
    for fname in [f"dml_robustness_{cat}.csv", f"dml_{cat}.csv", f"dml_results_{cat}.csv"]:
        p = os.path.join(STAGE9, "tables", fname)
        if not os.path.exists(p):
            p = os.path.join(STAGE9, fname)
        result["paths_checked"].append(p)
        if os.path.exists(p):
            result["found"] = True
            result["path"] = p
            result["data"] = pd.read_csv(p)
            break
    return result


diag_lines = []
diag_lines.append("DML Cross-Fitting and Inference Audit")
diag_lines.append("=" * 50)
diag_lines.append(f"Run date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
diag_lines.append("")

verify_rows = []

for cat in CATEGORIES:
    diag_lines.append(f"\n{cat.upper()}")
    diag_lines.append("-" * 30)

    r9 = load_stage9(cat)

    if r9["found"]:
        df9 = r9["data"]
        diag_lines.append(f"Stage9 results found: {r9['path']}")
        diag_lines.append(f"  Columns: {list(df9.columns)}")
        diag_lines.append(f"  Rows: {len(df9)}")
        diag_lines.append(f"  Content preview:\n{df9.to_string(index=False)}")

        # Check for grouped cross-fitting indicators
        has_grouped = any(c in df9.columns for c in
                         ["grouped_cv", "cluster_cv", "fold_type", "cv_type"])
        has_clustered_se = any(c in df9.columns for c in
                               ["cluster_se", "vcov", "se_type", "inference_type"])
        has_fold_sensitivity = len(df9) > 1  # multiple rows suggest fold sensitivity

        diag_lines.append(f"\n  CHECK: Grouped cross-fitting column present: {has_grouped}")
        diag_lines.append(f"  CHECK: Clustered SE column present: {has_clustered_se}")
        diag_lines.append(f"  CHECK: Fold sensitivity rows (>1): {has_fold_sensitivity}")

        # Look for fold count column
        fold_col = next((c for c in ["n_folds", "K", "folds", "n_splits"] if c in df9.columns), None)
        if fold_col:
            diag_lines.append(f"  Fold counts tested: {df9[fold_col].unique().tolist()}")

        # W5 coefficient across specs
        coef_col = next((c for c in ["theta", "ate", "coef", "W5", "estimate"] if c in df9.columns), None)
        if coef_col:
            diag_lines.append(f"  ATE estimates ({coef_col}):")
            for _, row in df9.iterrows():
                diag_lines.append(f"    {row.to_dict()}")

        verify_rows.append({
            "category": cat,
            "stage9_found": True,
            "path": r9["path"],
            "has_grouped_cv_column": has_grouped,
            "has_clustered_se_column": has_clustered_se,
            "n_rows": len(df9),
            "columns": ", ".join(df9.columns.tolist()),
        })
    else:
        diag_lines.append("Stage9 results NOT found. Paths checked:")
        for p in r9["paths_checked"]:
            diag_lines.append(f"  {p}")
        diag_lines.append("  -> Will run DML from scratch with grouped cross-fitting.")
        verify_rows.append({
            "category": cat, "stage9_found": False,
            "has_grouped_cv_column": False, "has_clustered_se_column": False,
        })

# ── Step 2: Implement grouped cross-fitting DML ───────────────────────────

print("\nChecking for econml/sklearn availability...")
try:
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold, KFold
    sklearn_ok = True
    print("  scikit-learn: OK")
except ImportError:
    sklearn_ok = False
    print("  scikit-learn: NOT FOUND — pip install scikit-learn")

try:
    from econml.dml import CausalForestDML, LinearDML
    econml_ok = True
    print("  econml: OK")
except ImportError:
    econml_ok = False
    print("  econml: NOT FOUND — pip install econml")

import pyarrow.parquet as pq

rng = np.random.default_rng(RANDOM_SEED)

fold_sensitivity_rows = []

if sklearn_ok:
    for cat in CATEGORIES:
        diag_lines.append(f"\nRunning DML for {cat.upper()} (sample frac={SAMPLE_FRAC})...")
        fpath = os.path.join(STAGE2, f"rms_{cat}_analysis.parquet")
        if not os.path.exists(fpath):
            diag_lines.append(f"  Parquet not found: {fpath}")
            continue

        pf_file = pq.ParquetFile(fpath)

        # Sample stores
        stores = set()
        for batch in pf_file.iter_batches(100_000, columns=["store_code_uc"]):
            stores.update(batch.column("store_code_uc").to_pylist())
        sampled_stores = set(rng.choice(sorted(stores),
                                        max(1, int(len(stores) * SAMPLE_FRAC)),
                                        replace=False))

        batches = []
        for batch in pf_file.iter_batches(200_000):
            pdf = batch.to_pandas()
            batches.append(pdf[pdf["store_code_uc"].isin(sampled_stores)])
        panel = pd.concat(batches, ignore_index=True)
        panel = panel.dropna(subset=["log_units", "W5"])
        diag_lines.append(f"  Panel rows: {len(panel):,}")

        # Feature matrix X
        # Use numerical features available in the consolidated schema
        feature_cols = [c for c in ["discount_depth", "regular_price", "unit_price",
                                     "week_of_year", "panel_year"]
                        if c in panel.columns]
        if not feature_cols:
            diag_lines.append("  No feature columns found — skip DML")
            continue

        panel_clean = panel.dropna(subset=feature_cols + ["W5", "log_units"])
        X = panel_clean[feature_cols].values
        Y = panel_clean["log_units"].values
        W = panel_clean["W5"].values
        store_ids = panel_clean["store_code_uc"].values

        # Map store IDs to integer groups for GroupKFold
        store_idx = pd.Categorical(store_ids).codes  # 0, 1, 2, ...
        n_stores  = len(np.unique(store_idx))
        diag_lines.append(f"  n_stores sampled: {n_stores}")
        diag_lines.append(f"  n_obs: {len(Y):,}")
        diag_lines.append(f"  W=1 fraction: {W.mean():.3f}")

        for K in FOLD_COUNTS:
            if K > n_stores:
                diag_lines.append(f"  K={K} skipped (only {n_stores} stores)")
                continue

            # ── Grouped cross-fitting (correct approach) ──────────────────────
            # GroupKFold ensures all obs from one store are in same fold
            gkf = GroupKFold(n_splits=K)

            # Partial-out W (propensity) using within-fold models
            W_res = np.empty_like(W, dtype=float)
            Y_res = np.empty_like(Y, dtype=float)

            for train_idx, test_idx in gkf.split(X, Y, groups=store_idx):
                X_train, X_test = X[train_idx], X[test_idx]
                W_train, W_test = W[train_idx], W[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                # Propensity model (linear logistic approximation via LassoCV)
                # With binary W, use logistic for proper probability estimates
                # For simplicity here, use OLS (first-stage in linear DML)
                from sklearn.linear_model import Ridge
                m_W = Ridge(alpha=1.0).fit(X_train, W_train)
                m_Y = Ridge(alpha=1.0).fit(X_train, Y_train)

                W_res[test_idx] = W_test - m_W.predict(X_test)
                Y_res[test_idx] = Y_test - m_Y.predict(X_test)

            # Second stage: regress Y_res on W_res (no constant — already partialled)
            # Clustered SE at store level
            theta_ols = np.dot(W_res, Y_res) / np.dot(W_res, W_res)

            # Clustered sandwich SE
            n = len(Y_res)
            score = W_res * (Y_res - theta_ols * W_res)   # n-vector of scores
            scores_by_store = pd.Series(score).groupby(pd.Series(store_idx)).sum().values
            meat = np.sum(scores_by_store ** 2)
            bread = np.dot(W_res, W_res)
            se_clustered = np.sqrt(meat / (bread ** 2))
            t_stat = theta_ols / se_clustered

            diag_lines.append(f"\n  K={K} grouped CV:")
            diag_lines.append(f"    theta (ATE) = {theta_ols:.4f}")
            diag_lines.append(f"    SE (store-clustered) = {se_clustered:.4f}")
            diag_lines.append(f"    t = {t_stat:.3f}")
            diag_lines.append(f"    95% CI: [{theta_ols - 1.96*se_clustered:.4f}, "
                              f"{theta_ols + 1.96*se_clustered:.4f}]")

            fold_sensitivity_rows.append({
                "category": cat, "K_folds": K, "cv_type": "grouped_store",
                "theta": round(theta_ols, 4), "se_clustered": round(se_clustered, 4),
                "t_stat": round(t_stat, 3),
                "ci_low": round(theta_ols - 1.96*se_clustered, 4),
                "ci_high": round(theta_ols + 1.96*se_clustered, 4),
                "n_obs": n, "n_stores": n_stores,
            })

        # ── Check stability across K ─────────────────────────────────────────
        cat_rows = [r for r in fold_sensitivity_rows if r["category"] == cat]
        if len(cat_rows) > 1:
            thetas = [r["theta"] for r in cat_rows]
            theta_range = max(thetas) - min(thetas)
            diag_lines.append(f"\n  ATE stability across K: range = {theta_range:.4f}")
            diag_lines.append(f"  (< 0.02 = stable; > 0.05 = investigate)")

else:
    diag_lines.append("\nsklearn not available — DML re-estimation skipped.")
    diag_lines.append("Install with: pip install scikit-learn econml")

# ── Step 3: Summary ───────────────────────────────────────────────────────

verify_df = pd.DataFrame(verify_rows)
fold_df   = pd.DataFrame(fold_sensitivity_rows)

verify_df.to_csv(os.path.join(OUT_DIR, "task12_dml_verification.csv"), index=False)
fold_df.to_csv(os.path.join(OUT_DIR, "task12_dml_fold_sensitivity.csv"), index=False)

diag_text = "\n".join(diag_lines)
with open(os.path.join(OUT_DIR, "task12_dml_diagnostic.txt"), "w") as f:
    f.write(diag_text)

print(diag_text)

if not fold_df.empty:
    print("\n--- Fold sensitivity summary ---")
    print(fold_df.to_string(index=False))

print(f"\nOutputs written to {OUT_DIR}")
print("""
INTERPRETATION NOTES:
  Grouped cross-fitting:
    • All obs from one store in the same fold → nuisance estimator truly out-of-sample
    • Without grouping: same-store residuals can leak into model, inflating precision
    • Key check: theta stable across K=3,5,10 → DML estimate is robust

  Clustered SE:
    • Store-level clustering is correct: promotion timing is DMA/store-level
    • HC-robust SE underestimates variance if store-level shocks are correlated
    • Check: SE(store-clustered) > SE(HC-robust) is expected and correct

  If existing stage9 used standard KFold without grouping:
    • Re-run with GroupKFold and report both for comparison
    • Expected direction: grouped CV has slightly larger SE, possibly different theta
    • Document the difference in IS Report v4 Section 6 (DML robustness)
""")
print("Task 12 complete.")
