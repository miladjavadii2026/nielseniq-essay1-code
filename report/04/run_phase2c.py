"""
run_phase2c.py
==============
Phase 2C master orchestrator — runs all 12 analysis tasks in sequence.

USAGE
-----
  python run_phase2c.py                    # run all tasks
  python run_phase2c.py --tasks 1 3 5      # run specific tasks only
  python run_phase2c.py --skip 8 9         # skip HMS tasks (if no HMS data)
  python run_phase2c.py --dry-run          # print plan without executing

TASK LIST
---------
  01  Zero-sale audit (zero_sales_audit)
  02  Pooled category × depth interaction model  [stage11 — already run]
  03  Adjacent-bin, monotonicity, curvature tests
  04  Calendar-week FE added to event study
  05  Bootstrap CIs for offset statistic
  06  Long post-promotion windows (16-week)
  07  Promotion duration separation (1-wk vs. multi-wk)
  08  Household FE quantity models (HMS)
  09  Interpurchase / hazard models (HMS)
  10  Clearance proxies near UPC exit
  11  Causal-forest heterogeneity (RATE/GATES)
  12  DML grouped cross-fitting and clustered inference audit

NOTE: Task 02 corresponds to stage11_pooled_interaction.py (already in
the report folder). It is listed for completeness; pass --skip 2 to skip.

DESIGN PRINCIPLES
------------------
- Each task is run in an isolated subprocess to prevent memory leaks
- Output goes to 06_results/task{NN}_*/
- Failures are logged but do not abort the run (other tasks continue)
- A final summary report is written to 06_results/phase2c_summary.txt
- SAMPLE_FRAC applies to all RMS parquet reads (laptop-safe)

ENVIRONMENT
-----------
Required packages:
  pip install pyarrow pyfixest econml scikit-learn statsmodels --break-system-packages

All source parquet files are READ-ONLY under:
  F:\\Phd in economics\\Term5\\independent\\report\\figures and resulrs\\06_results\\stage2_variables\\
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUT_ROOT   = (Path(r"F:\Phd in economics\Term5\independent\report\figures and resulrs")
              / "06_results")

TASKS = {
    1:  ("task01_zero_sales_audit.py",       "Zero-sale audit"),
    2:  ("stage11_pooled_interaction.py",    "Pooled category × depth interaction [STAGE11]"),
    3:  ("task03_bin_monotonicity_tests.py", "Adjacent-bin monotonicity tests"),
    4:  ("task04_event_study_calweek_fe.py", "Calendar-week FE event study"),
    5:  ("task05_offset_bootstrap.py",       "Bootstrap CIs for offset"),
    6:  ("task06_long_window_event.py",      "Long post-promotion windows"),
    7:  ("task07_promo_duration.py",         "Promotion duration separation"),
    8:  ("task08_household_fe_quantity.py",  "Household FE quantity models [HMS]"),
    9:  ("task09_interpurchase_hazard.py",   "Interpurchase / hazard models [HMS]"),
    10: ("task10_clearance_proxies.py",      "Clearance proxies near UPC exit"),
    11: ("task11_cf_heterogeneity.py",       "Causal-forest heterogeneity"),
    12: ("task12_dml_grouped_cf.py",         "DML grouped cross-fitting audit"),
}

# Stage11 is in the same folder but named differently
TASK2_ALSO_CHECK = SCRIPT_DIR / "stage11_pooled_interaction.py"

# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2C orchestrator")
    p.add_argument("--tasks", nargs="+", type=int, default=None,
                   help="Run only these task numbers (e.g. --tasks 1 3 5)")
    p.add_argument("--skip", nargs="+", type=int, default=[],
                   help="Skip these task numbers (e.g. --skip 8 9 for no HMS)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan but do not execute")
    p.add_argument("--timeout", type=int, default=3600,
                   help="Per-task timeout in seconds (default 3600 = 1 hour)")
    return p.parse_args()


# ── Runner ──────────────────────────────────────────────────────────────────

def run_task(task_num: int, script_name: str, timeout: int, dry_run: bool) -> dict:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        # Try task02 fallback
        if task_num == 2 and TASK2_ALSO_CHECK.exists():
            script_path = TASK2_ALSO_CHECK
        else:
            return {"task": task_num, "script": script_name,
                    "status": "SKIPPED_NOT_FOUND", "elapsed": 0.0}

    if dry_run:
        print(f"  [DRY RUN] Would run: python {script_path.name}")
        return {"task": task_num, "script": script_name,
                "status": "DRY_RUN", "elapsed": 0.0}

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        status  = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = timeout
        status  = f"TIMEOUT (>{timeout}s)"
    except Exception as e:
        elapsed = time.time() - t0
        status  = f"ERROR: {e}"

    return {"task": task_num, "script": script_name, "status": status, "elapsed": round(elapsed, 1)}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Determine which tasks to run
    task_nums = args.tasks if args.tasks else sorted(TASKS.keys())
    task_nums = [t for t in task_nums if t not in args.skip]
    task_nums = [t for t in task_nums if t in TASKS]

    print("=" * 65)
    print("Phase 2C — Analysis Tasks Orchestrator")
    print("=" * 65)
    print(f"  Script dir: {SCRIPT_DIR}")
    print(f"  Output root: {OUT_ROOT}")
    print(f"  Python: {sys.executable}")
    print(f"  Tasks to run: {task_nums}")
    if args.skip:
        print(f"  Skipping: {args.skip}")
    if args.dry_run:
        print("  MODE: DRY RUN (no execution)")
    print()

    # Pre-flight: check scripts exist
    print("Pre-flight check:")
    for n in task_nums:
        script  = TASKS[n][0]
        exists  = (SCRIPT_DIR / script).exists() or (n == 2 and TASK2_ALSO_CHECK.exists())
        status  = "✓" if exists else "✗ NOT FOUND"
        print(f"  Task {n:02d}: {script:45s}  {status}")
    print()

    if not args.dry_run:
        confirm = input("Proceed? [Y/n] ").strip().lower()
        if confirm == "n":
            print("Aborted.")
            return

    # Run tasks
    print("\nRunning tasks...")
    results = []
    total_start = time.time()

    for n in task_nums:
        script, label = TASKS[n]
        print(f"\n{'='*50}")
        print(f"Task {n:02d}: {label}")
        print(f"  Script: {script}")
        print(f"  Start:  {time.strftime('%H:%M:%S')}")
        print()

        r = run_task(n, script, args.timeout, args.dry_run)
        results.append(r)

        print(f"\n  → {r['status']}  (elapsed: {r['elapsed']:.0f}s)")

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 65)
    print("Phase 2C Complete")
    print("=" * 65)
    print(f"Total elapsed: {total_elapsed/60:.1f} min")
    print()
    print(f"{'Task':>4}  {'Script':45s}  {'Status':30s}  {'Elapsed':>8}")
    print("-" * 95)

    summary_lines = ["Phase 2C Summary", "=" * 50, ""]
    ok_count, fail_count, skip_count = 0, 0, 0

    for r in results:
        tag = "✓" if r["status"] == "OK" else ("⚡" if "DRY" in r["status"] else "✗")
        row = f"  {r['task']:>2}  {r['script']:45s}  {r['status']:30s}  {r['elapsed']:>6.0f}s"
        print(f" {tag} {row}")
        summary_lines.append(f"{tag} Task {r['task']:02d}: {r['script']} — {r['status']} ({r['elapsed']}s)")

        if r["status"] == "OK":
            ok_count += 1
        elif "SKIPPED" in r["status"] or "DRY" in r["status"]:
            skip_count += 1
        else:
            fail_count += 1

    print("-" * 95)
    print(f"  OK: {ok_count}   Failed: {fail_count}   Skipped/dry: {skip_count}")
    print()

    summary_lines += [
        "", f"Total elapsed: {total_elapsed/60:.1f} min",
        f"OK: {ok_count}  Failed: {fail_count}  Skipped: {skip_count}",
        "",
        "OUTPUT DIRECTORIES:",
    ]
    for n in task_nums:
        out_dir = OUT_ROOT / f"task{n:02d}_*"
        summary_lines.append(f"  Task {n:02d}: {out_dir}")

    summary_lines += [
        "",
        "NEXT STEPS:",
        "  1. Review per-task CSV outputs in 06_results/task*/",
        "  2. Update IS_Report_Revised_v4.docx with confirmed results",
        "  3. For tasks 8/9 (HMS): verify HMS parquet exists and rerun",
        "  4. For task 11 (CF): re-run stage6/8 with SAVE_FULL_CATE=True for RATE test",
        "  5. For task 12 (DML): compare grouped vs. standard CV estimates",
    ]

    summary_text = "\n".join(summary_lines)
    summary_path = OUT_ROOT / "phase2c_summary.txt"
    try:
        summary_path.write_text(summary_text)
        print(f"Summary written to: {summary_path}")
    except Exception as e:
        print(f"Could not write summary: {e}")


if __name__ == "__main__":
    main()
