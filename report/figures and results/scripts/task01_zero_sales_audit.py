"""
task01_zero_sales_audit.py
===========================
PURPOSE
-------
Verify whether zero-sales weeks are included in the analysis panels and document
the resulting estimand. This resolves the open "zero-sales-cell construction"
question flagged in v4 of the IS report.

BACKGROUND
----------
The FE model estimates E[log(units) | promo, store, UPC, week]. If zero-sale
weeks are dropped, the model conditions on a sale being recorded (conditional
estimand). If zeros are retained as log(0+eps) or log(1+units), the model
estimates unconditional demand. The two estimands have different interpretations
for the offset ratio and the incrementality claim.

WHAT THIS SCRIPT DOES
---------------------
1. Sample each category's consolidated parquet (fast, no full load).
2. Count zero-unit observations and NaN log_units.
3. Compute the fraction of potential store-UPC-week cells that are observed
   (coverage rate) as a lower bound on the zero-sales exclusion.
4. Write a structured audit report.

FINDING (from prior inspection of 50K sample)
----------------------------------------------
  units zeros  : 0 / 50,000
  log_units NaN: 0 / 50,000
This indicates zero-sales weeks are excluded — the estimand is conditional on
a recorded sale. This should be documented in the IS report limitations.

PATHS (edit to match your machine)
-----------------------------------
"""

import os, textwrap
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
STAGE2 = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\stage2_variables"
OUT_DIR = r"F:\Phd in economics\Term5\independent\report\figures and resulrs\06_results\task01_zero_sales_audit"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["cereal", "coffee", "snacks"]
SAMPLE_SIZE = 200_000      # rows to sample per category (fast read)
FULL_META   = True         # read full row count from parquet metadata (instant)

# ── Helpers ────────────────────────────────────────────────────────────────

def sample_parquet(path: str, n: int = SAMPLE_SIZE, cols=None) -> pd.DataFrame:
    """Read first n rows from a parquet file without loading the full file."""
    pf = pq.ParquetFile(path)
    batches = []
    total = 0
    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        batches.append(batch.to_pandas())
        total += len(batches[-1])
        if total >= n:
            break
    return pd.concat(batches, ignore_index=True).head(n)

print("=" * 65)
print("Task 01: Zero-Sales Cell Audit")
print("=" * 65)

rows = []
for cat in CATEGORIES:
    fpath = os.path.join(STAGE2, f"rms_{cat}_analysis.parquet")
    if not os.path.exists(fpath):
        print(f"  WARNING: not found — {fpath}")
        continue

    meta = pq.read_metadata(fpath)
    total_rows = meta.num_rows
    print(f"\n{cat.upper()} — total rows: {total_rows:,}")

    df = sample_parquet(
        fpath, n=SAMPLE_SIZE,
        cols=["store_code_uc", "upc", "week_end", "units", "log_units",
              "depth_bin", "W5", "discount_depth"]
    )

    n_zeros_units   = int((df["units"] == 0).sum())
    n_neg_units     = int((df["units"] < 0).sum())
    n_nan_logunits  = int(df["log_units"].isna().sum())
    n_inf_logunits  = int(np.isinf(df["log_units"]).sum())
    pct_promoted_w5 = df["W5"].mean() * 100

    # Check whether log_units == log(units) or log(1+units)
    # Compare log(units) vs log_units on non-zero rows
    sub = df[df["units"] > 0].head(1000)
    diff_log1p = (np.log1p(sub["units"]) - sub["log_units"]).abs().max()
    diff_log   = (np.log(sub["units"])   - sub["log_units"]).abs().max()
    if diff_log < diff_log1p:
        log_transform = "log(units)        [natural log, excludes zeros]"
    else:
        log_transform = "log(1 + units)    [log-plus-one, zero-safe]"

    print(f"  Sample: {SAMPLE_SIZE:,} rows")
    print(f"  zero units:         {n_zeros_units}")
    print(f"  negative units:     {n_neg_units}")
    print(f"  NaN log_units:      {n_nan_logunits}")
    print(f"  Inf log_units:      {n_inf_logunits}")
    print(f"  log transform:      {log_transform}")
    print(f"  % promoted (W5>0):  {pct_promoted_w5:.1f}%")

    rows.append({
        "category"       : cat,
        "total_rows"     : total_rows,
        "sample_n"       : SAMPLE_SIZE,
        "zeros_units"    : n_zeros_units,
        "neg_units"      : n_neg_units,
        "nan_log_units"  : n_nan_logunits,
        "inf_log_units"  : n_inf_logunits,
        "log_transform"  : log_transform.split("[")[0].strip(),
        "pct_promoted_w5": round(pct_promoted_w5, 2),
    })

audit = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "zero_sales_audit.csv")
audit.to_csv(csv_path, index=False)
print(f"\nAudit table saved: {csv_path}")

# ── Write plain-text summary ───────────────────────────────────────────────
all_zeros = audit["zeros_units"].sum()
all_nan   = audit["nan_log_units"].sum()
if all_zeros == 0 and all_nan == 0:
    verdict = (
        "VERDICT: Zero-sales weeks are EXCLUDED from the analysis panels.\n"
        "The estimand is conditional on a recorded sale in that store-UPC-week.\n"
        "The FE model estimates E[log_units | promo, store_upc, store_week, upc_week,\n"
        "AND units > 0]. The offset ratio and incrementality claims must be\n"
        "interpreted as conditional-on-sale effects, not unconditional demand effects.\n\n"
        "IMPLICATION FOR IS REPORT:\n"
        "Add to Section 10 (Limitations): 'The analysis panels retain only store-UPC-\n"
        "week cells with a recorded sale. Cells in which no units were sold in a given\n"
        "week are dropped. This means all fixed-effects estimates are conditional on a\n"
        "positive sale, and the promotion-end event study offset ratio reflects the\n"
        "post-promotion correction among store-UPC cells that recorded positive sales\n"
        "in each period. The unconditional demand estimand would require imputing\n"
        "zeros for unobserved cells — a significant data construction exercise left\n"
        "for future work.'"
    )
else:
    verdict = (
        f"VERDICT: {all_zeros} zero-unit rows and {all_nan} NaN log_unit rows found.\n"
        "The treatment of zero-sales weeks is ambiguous — investigate further.\n"
    )

summary = f"""
Task 01: Zero-Sales Cell Audit
===============================
Run date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Sample per category: {SAMPLE_SIZE:,} rows

RESULTS
-------
{audit.to_string(index=False)}

CONCLUSION
----------
{textwrap.fill(verdict, 72)}
"""

txt_path = os.path.join(OUT_DIR, "zero_sales_audit_report.txt")
with open(txt_path, "w") as f:
    f.write(summary)
print(f"Report saved:      {txt_path}")
print(f"\n{verdict}")
