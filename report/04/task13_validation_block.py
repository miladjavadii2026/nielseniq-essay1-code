from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(r"F:\Phd in economics\Term5\independent")
RESULTS = ROOT / "report" / "figures and resulrs" / "06_results"
OUT = RESULTS / "task13_validation_block"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["cereal", "coffee", "snacks"]
BASELINE_TIMES = [-8, -7, -6, -5]
DENOM_TIMES = [-4, -3, -2, -1, 0]
HORIZONS = [4, 8, 12]


def fmt_times(times: list[int]) -> str:
    return ",".join(f"t{v:+d}" for v in times)


def offset_reconciliation() -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        path = RESULTS / "stage5_event_study" / "rms_fe" / f"event_study_fe_{cat}.csv"
        df = pd.read_csv(path)
        coef = dict(zip(df["event_time"], df["coef"]))
        baseline_level = float(np.mean([coef[t] for t in BASELINE_TIMES]))
        denom_adjusted = float(sum(coef[t] - baseline_level for t in DENOM_TIMES))
        denom_raw = float(sum(coef[t] for t in DENOM_TIMES))

        for h in HORIZONS:
            num_times = list(range(1, h + 1))
            num_adjusted = float(sum(coef[t] - baseline_level for t in num_times))
            num_raw = float(sum(coef[t] for t in num_times))
            rows.append(
                {
                    "category": cat,
                    "horizon_weeks": h,
                    "baseline_event_times": fmt_times(BASELINE_TIMES),
                    "baseline_level": round(baseline_level, 6),
                    "numerator_event_times": fmt_times(num_times),
                    "denominator_event_times": fmt_times(DENOM_TIMES),
                    "numerator_sum_adjusted": round(num_adjusted, 6),
                    "denominator_sum_adjusted": round(denom_adjusted, 6),
                    "signed_ratio_adjusted": round(num_adjusted / denom_adjusted, 6),
                    "absolute_ratio_adjusted": round(abs(num_adjusted / denom_adjusted), 6),
                    "numerator_sum_raw": round(num_raw, 6),
                    "denominator_sum_raw": round(denom_raw, 6),
                    "note": "Adjusted sums subtract the mean coefficient over t=-8,-7,-6,-5 from each numerator/denominator event-time coefficient.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "task13_offset_reconciliation.csv", index=False)
    return out


def pairwise_category_contrasts() -> pd.DataFrame:
    fe = {}
    for cat in CATEGORIES:
        path = RESULTS / "stage4_fe_models" / "tables" / f"fe_seven_bin_{cat}.csv"
        df = pd.read_csv(path).set_index("term")
        fe[cat] = df

    rows = []
    for bin_name in ["D6", "D7"]:
        for a, b in [("cereal", "coffee"), ("cereal", "snacks"), ("snacks", "coffee")]:
            diff = float(fe[a].loc[bin_name, "estimate"] - fe[b].loc[bin_name, "estimate"])
            se = float(np.sqrt(fe[a].loc[bin_name, "std_error"] ** 2 + fe[b].loc[bin_name, "std_error"] ** 2))
            z = diff / se
            p = 2 * stats.norm.sf(abs(z))
            rows.append(
                {
                    "bin": bin_name,
                    "contrast": f"{a} - {b}",
                    "estimate_a": round(float(fe[a].loc[bin_name, "estimate"]), 6),
                    "estimate_b": round(float(fe[b].loc[bin_name, "estimate"]), 6),
                    "difference": round(diff, 6),
                    "se_conservative": round(se, 6),
                    "z_stat": round(z, 3),
                    "p_value_two_sided": round(p, 6),
                    "significant_5pct": bool(p < 0.05),
                    "inference_note": "Conservative direct contrast from separately estimated category FE coefficients; exact pooled Wald contrast with full covariance remains pending.",
                }
            )

        # Approximate joint equality across the three categories at this bin.
        for ref in ["cereal"]:
            diffs = []
            for other in ["coffee", "snacks"]:
                diff = float(fe[ref].loc[bin_name, "estimate"] - fe[other].loc[bin_name, "estimate"])
                se = float(np.sqrt(fe[ref].loc[bin_name, "std_error"] ** 2 + fe[other].loc[bin_name, "std_error"] ** 2))
                diffs.append((diff / se) ** 2)
            chi2 = float(sum(diffs))
            p_joint = float(stats.chi2.sf(chi2, df=2))
            rows.append(
                {
                    "bin": bin_name,
                    "contrast": "joint category equality, reference cereal",
                    "estimate_a": np.nan,
                    "estimate_b": np.nan,
                    "difference": np.nan,
                    "se_conservative": np.nan,
                    "z_stat": round(chi2, 3),
                    "p_value_two_sided": round(p_joint, 6),
                    "significant_5pct": bool(p_joint < 0.05),
                    "inference_note": "Approximate chi-square test from conservative separate-model contrasts; exact pooled covariance Wald test remains pending.",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "task13_pairwise_category_contrasts_conservative.csv", index=False)
    return out


def adjacent_bin_table() -> pd.DataFrame:
    src = RESULTS / "task03_bin_tests" / "task03_adjacent_bin_tests.csv"
    df = pd.read_csv(src)
    df = df.rename(
        columns={
            "se_diff_conservative": "standard_error",
            "p_value": "p_value_two_sided",
        }
    )
    df["hypothesis"] = "two-sided H0: beta_hi - beta_lo = 0; monotone direction indicated by diff > 0"
    df.to_csv(OUT / "task13_adjacent_bin_test_table.csv", index=False)

    sign_rows = []
    for cat, g in df.groupby("category"):
        n_pos = int((g["diff"] > 0).sum())
        n = int(len(g))
        sign_p_one_sided = stats.binomtest(n_pos, n, p=0.5, alternative="greater").pvalue
        sign_rows.append(
            {
                "category": cat,
                "positive_steps": n_pos,
                "total_steps": n,
                "one_sided_sign_test_p": round(float(sign_p_one_sided), 6),
                "hypothesis": "H1: more positive adjacent steps than expected under random signs",
            }
        )
    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(OUT / "task13_adjacent_sign_tests.csv", index=False)
    return df


if __name__ == "__main__":
    off = offset_reconciliation()
    pair = pairwise_category_contrasts()
    adj = adjacent_bin_table()
    print("Offset reconciliation")
    print(off.to_string(index=False))
    print("\nPairwise category contrasts")
    print(pair.to_string(index=False))
    print("\nAdjacent-bin table")
    print(adj.to_string(index=False))
    print(f"\nSaved outputs under: {OUT}")
