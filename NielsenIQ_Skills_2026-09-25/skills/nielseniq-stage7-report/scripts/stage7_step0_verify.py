#!/usr/bin/env python3
"""Stage 7, Step 0: verify what is complete and confirm module codes BEFORE
any report packaging. Reads only. Never writes under G:.

Usage:
  python stage7_step0_verify.py --base "E:\\NielsenIQ_Phase2" [--out DIR] [--rowcounts]

What it checks
  1. Stage completion from the pipeline's own marker files (.stage_N_complete,
     .itemN_complete) plus a file scan of every 06_results stage folder.
     A marker with "method": "forced" or a "warning" key is reported as
     COMPLETE (FORCED); the report must not cite that stage's validation as passed.
  2. Interrupted atomic writes (*.partial.parquet) anywhere under 06_results.
  3. Module codes: build-script constants (GROUP, MODULE) versus the corrected
     movement-file map (official group/module pairs) versus document claims
     (locked_stages_godmode_v2_3.md, FINAL_RERUN_v12/00_analysis_manifest.csv).
     Group codes (1005 cereal, 1006 coffee, 1507 snacks) and module codes
     (1344, 1463, 1323) are different identifiers; the check names which is which.
Outputs: step0_verification_report.csv and .md in --out.
Exit code 1 on any CONFLICT, partial file, or unreadable marker; 0 otherwise.
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path

CATS = ["cereal", "snacks", "coffee"]          # locked order: Cereal, Snacks, Coffee
# (id, label, folder under 06_results, marker paths under 06_results, expected categories)
STAGES = [
    ("1",   "Data foundation and manifest",          "stage1_manifest",            ["stage1_manifest/.stage_1_complete"], CATS),
    ("2",   "Analysis variables",                    "stage2_variables",           ["stage2_variables/.stage_2_complete"], CATS),
    ("3",   "Descriptives",                          "stage3_descriptives",        ["stage3_descriptives/.stage_3_complete"], CATS),
    ("4",   "Seven-bin fixed effects",               "stage4_fe_models",           ["stage4_fe_models/.stage_4_complete"], CATS),
    ("5",   "Event study, descriptive (rms)",        "stage5_event_study/rms",     ["stage5_event_study/.stage_5_complete"], CATS),
    ("5FE", "Event study, FE + offset share (rms_fe)","stage5_event_study/rms_fe", ["stage5_event_study/rms_fe/.item1_complete", "stage5_event_study/rms_fe/.item2_complete"], CATS),
    ("5H",  "Household matching (hms)",              "stage5_event_study/hms",     ["stage5_event_study/hms/.item3_complete", "stage5_event_study/hms/.item4_complete"], CATS),
    ("6",   "Causal-forest pilot (cereal, 2015)",    "stage6_causal_forest",       ["stage6_causal_forest/.stage_6_complete", "stage6_causal_forest/.item6_complete", "stage6_causal_forest/.item7_complete"], ["cereal"]),
    ("7",   "Report",                                "stage7_report",              ["stage7_report/.stage_7_complete"], []),
    ("8",   "Scaled causal forest, all categories",  "stage8_scale_panel",         ["stage8_scale_panel/.item8_complete"], CATS),
    ("9",   "DML robustness",                        "stage9_dml_robustness",      ["stage9_dml_robustness/.item9_complete"], CATS),
    ("10",  "Cereal divergence",                     "stage10_cereal_divergence",  ["stage10_cereal_divergence/.item10_complete"], ["cereal"]),
    ("11",  "Pooled category-by-depth interaction",  "stage11_pooled_interaction", ["stage11_pooled_interaction/.item11_complete"], CATS),
]
SKIP_DIRS = {"_tmp_years", "__pycache__"}
LOAD_FIRST = [
    "milad-dissertation skill (citation registry, architecture, code rules)",
    "research-report-qa skill (build, split, verify, verdict loop, data map)",
    "docx skill (Step 6 build) and pdf skill (Step 5 results-records package)",
    "anti-ai-writing-style.md and register-switching-guide.md (Academic/Precise)",
    "most recent NielsenIQ Phase 2 context transfer file (live status)",
]


def on_g(p: Path) -> bool:
    s = str(p).replace("/", "\\").upper()
    return s.startswith("G:") or "\\G\\NIELSENIQ" in s or s.startswith("\\MNT\\USER-DATA\\UPLOADS\\G")


def read_marker(p: Path) -> dict:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            txt = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            m = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", txt)
            return {"raw": txt.strip(), "completed_at": m.group(0) if m else ""}
    return {"error": "undecodable"}


def scan(folder: Path):
    files, partial = [], []
    if not folder.exists():
        return files, partial
    for f in folder.rglob("*"):
        if not f.is_file() or f.name.startswith("."):
            continue
        if any(part in SKIP_DIRS for part in f.relative_to(folder).parts):
            continue
        (partial if f.name.endswith(".partial.parquet") else files).append(f)
    return files, partial


def rowcount(p: Path):
    try:
        import duckdb
        return duckdb.sql(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0]
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__})"


def stage_rows(results: Path, want_rowcounts: bool):
    rows, problems = [], []
    for sid, label, folder, markers, cats in STAGES:
        fdir = results / folder
        files, partial = scan(fdir)
        if partial:
            problems.append(f"Stage {sid}: {len(partial)} interrupted write(s): " + ", ".join(x.name for x in partial))
        marks = {}
        for m in markers:
            mp = results / m
            if mp.exists():
                marks[m] = read_marker(mp)
                if "error" in marks[m]:
                    problems.append(f"Stage {sid}: unreadable marker {m}")
        primary = next(iter(marks.values()), None)
        completed_at = (primary or {}).get("completed_at", "")
        forced = any(("warning" in v) or (v.get("method") == "forced") for v in marks.values())
        marker_cats = set()
        for v in marks.values():
            marker_cats.update(c.lower() for c in v.get("categories", []))
            if v.get("category"):
                marker_cats.add(str(v["category"]).lower())
        targets = cats or ["(all)"]
        for cat in targets:
            matched = [f for f in files if cat in f.name.lower() or cat in str(f.relative_to(fdir)).lower()] if cat != "(all)" else files
            if not matched and len(cats) == 1 and cat in marker_cats:
                matched = files          # single-category stage with unlabelled files (W5/W10)
            if marks:
                if forced:
                    status = "COMPLETE (FORCED)"
                elif marker_cats and cat != "(all)" and cat not in marker_cats:
                    status = "NOT COVERED BY MARKER"
                else:
                    status = "COMPLETE (marker)"
            elif matched:
                status = "FILES PRESENT, NO MARKER"
            elif fdir.exists():
                status = "NOT STARTED (folder empty)"
            else:
                status = "NOT STARTED (no folder)"
            note = "; ".join(f"{k}={v[k]}" for v in marks.values() for k in ("warning", "method", "spec", "windows", "pilot_year", "years") if k in v)
            rc = ""
            if want_rowcounts and cat != "(all)":
                pq = [f for f in matched if f.suffix == ".parquet"]
                rc = "; ".join(f"{f.name}={rowcount(f)}" for f in pq[:4])
            rows.append(dict(stage=sid, label=label, folder=folder, category=cat, status=status,
                             completed_at=completed_at, markers=", ".join(marks) or "none",
                             files_total=len(files), files_matched=len(matched), partial_files=len(partial),
                             rowcounts=rc, note=note))
            if status.startswith("NOT COVERED") or status.startswith("FILES PRESENT"):
                problems.append(f"Stage {sid} / {cat}: {status}")
        if forced:
            problems.append(f"Stage {sid}: marker is FORCED ({note}). Do not cite this stage's validation as passed;"
                            " cite the per-year build audits (see STAGE 1 EVIDENCE) instead.")
    return rows, problems


def stage1_evidence(base: Path):
    """The Stage 1 marker can be forced. The real evidence is the per-year build audit JSON
    in 01_documentation (current file, not *_previous_failed_*) and 08_logs/validation_*.txt."""
    lines = ["STAGE 1 EVIDENCE (build audits and validation logs)"]
    doc = base / "01_documentation"
    for cat in CATS:
        cur = sorted(doc.glob(f"nielseniq_rms_{cat}_*_7d1_build_audit.json")) if doc.exists() else []
        cur = [f for f in cur if "_previous_failed_" not in f.name]
        cur += sorted(doc.glob(f"nielseniq_rms_*_{cat}_module*_7c1c_build_audit.json")) if doc.exists() else []   # 2014 cereal pilot
        years = sorted({m.group(1) for f in cur for m in [re.search(r"(\d{4})", f.name)] if m})
        failed = len(list(doc.glob(f"nielseniq_rms_{cat}_*_7d1_build_audit_previous_failed_*.json"))) if doc.exists() else 0
        lines.append(f"  {cat}: {len(cur)} current build audit(s), years {', '.join(years) or 'none'}; {failed} superseded failed-attempt audit(s)")
    logs = sorted((base / "08_logs").glob("validation_*.txt")) if (base / "08_logs").exists() else []
    lines.append("  validation logs: " + (", ".join(f.name for f in logs) or "none found in 08_logs"))
    return lines


def module_codes(base: Path):
    lines, conflict = ["MODULE CODE CHECK (Step 0, item 2)"], False
    mp = base / "03_inventory" / "nielseniq_rms_movement_file_map_corrected.csv"
    official = {}
    if mp.exists():
        with open(mp, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.DictReader(fh)
            need = {"product_group_code", "product_group_descr", "product_module_code", "product_module_descr"}
            if not need.issubset(rd.fieldnames or []):
                lines.append(f"  CANNOT CONFIRM: map lacks columns {sorted(need - set(rd.fieldnames or []))}")
            else:
                for r in rd:
                    official.setdefault(r["product_group_descr"].strip().lower(), {})[r["product_module_code"].strip()] = (
                        r["product_group_code"].strip(), r["product_module_descr"].strip(), r.get("official_pair_valid", "").strip().lower())
    else:
        lines.append(f"  CANNOT CONFIRM: map not found: {mp}")
    claims = {}
    lock = base / "00_context" / "locked_stages_godmode_v2_3.md"
    if lock.exists():
        for cat, code in re.findall(r"(Cereal|Coffee|Snacks)\s*\((\d{4})", lock.read_text(encoding="utf-8", errors="replace")):
            claims.setdefault(cat.lower(), set()).add(code)
    man = base / "FINAL_RERUN_v12" / "00_analysis_manifest.csv"
    if man.exists():
        for r in csv.reader(open(man, newline="", encoding="utf-8", errors="replace")):
            if len(r) >= 2 and r[0].endswith("_module"):
                claims.setdefault(r[0].replace("_module", "").lower(), set()).add(r[1].strip())
    for cat in CATS:
        script = base / f"step_7d1_build_{cat}_years.py"
        grp = mod = None
        if script.exists():
            txt = script.read_text(encoding="utf-8", errors="replace")
            g = re.search(r'^GROUP\s*=\s*"(\d+)"', txt, re.M); m = re.search(r'^MODULE\s*=\s*"(\d+)"', txt, re.M)
            grp, mod = (g.group(1) if g else None), (m.group(1) if m else None)
        off = official.get(cat, {})
        doc = sorted(claims.get(cat, set()))
        if mod is None:
            lines.append(f"  {cat}: build script GROUP/MODULE not found; document claims {doc or 'none'}. CANNOT CONFIRM.")
            conflict = True; continue
        pair_ok = bool(off) and mod in off and off[mod][0] == grp and off[mod][2] in ("true", "1", "")
        doc_ok = not doc or set(doc) == {mod}
        if not off:
            verdict = "UNVERIFIED (map unavailable)"; conflict = True
        elif not pair_ok:
            verdict = "CONFLICT: build group/module is not an official pair in the map"; conflict = True
        elif not doc_ok:
            verdict = f"CONFLICT: build uses {mod} but documents claim {doc}; correct the documents"; conflict = True
        else:
            verdict = "CONFIRMED (build script = official map = documents)"
        descr = off.get(mod, (None, "?", None))[1]
        lines.append(f"  {cat}: group {grp} / module {mod} ({descr}); documents claim {doc or 'none'}: {verdict}")
    lines.append("  Note: 1005/1006/1507 are product GROUP codes (folder level); 1344/1463/1323 are product MODULE codes"
                 " (file level). Any exhibit that prints a code must say which one it is.")
    return lines, conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help=r"project root, normally E:\NielsenIQ_Phase2 (G: mirror is read-only)")
    ap.add_argument("--out", help="output folder; required when --base is on G:")
    ap.add_argument("--rowcounts", action="store_true", help="parquet row counts via duckdb (metadata read)")
    ap.add_argument("--require", nargs="*", default=["1", "2", "3", "4", "5", "5FE", "5H", "6"],
                    help="stage ids that must be COMPLETE for exit 0 (default: the IS2 stages)")
    a = ap.parse_args()
    base = Path(a.base); results = base / "06_results"
    out = Path(a.out) if a.out else results / "stage7_report" / "step0"
    if on_g(out) or (on_g(base) and not a.out):
        sys.exit("Refusing to write under G:. Pass --out on E: or a local folder.")
    print("LOAD BEFORE DRAFTING:"); [print("  -", x) for x in LOAD_FIRST]
    rows, problems = stage_rows(results, a.rowcounts)
    print(f"\nSTAGE STATUS ({results})")
    for r in rows:
        print(f"  Stage {r['stage']:<4} {r['category']:<7} {r['status']:<28} files {r['files_matched']}/{r['files_total']}  {r['completed_at']}  {r['note']}")
    for r in rows:
        if r["stage"] in a.require and not r["status"].startswith("COMPLETE"):
            problems.append(f"Stage {r['stage']} / {r['category']}: required but {r['status']}")
    elines = stage1_evidence(base)
    print(); print("\n".join(elines))
    mlines, mconf = module_codes(base)
    print(); print("\n".join(mlines))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "step0_verification_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["# Step 0 verification report", "", f"Base: `{base}`", "", "| Stage | Category | Status | Completed | Files | Note |", "|---|---|---|---|---|---|"]
    md += [f"| {r['stage']} {r['label']} | {r['category']} | {r['status']} | {r['completed_at']} | {r['files_matched']}/{r['files_total']} | {r['note']} |" for r in rows]
    md += ["", "## Stage 1 evidence", ""] + [l.strip() for l in elines[1:]]
    md += ["", "## Module codes", ""] + [l.strip() for l in mlines[1:]] + ["", "## Problems", ""] + ([f"- {p}" for p in problems] or ["- none"])
    (out / "step0_verification_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWritten: {out / 'step0_verification_report.csv'} and .md")
    if problems or mconf:
        print("PROBLEMS:"); [print("  -", p) for p in problems]
        sys.exit(1)
    print("STEP 0 CLEAR: no conflicts, no partial writes, no uncovered categories.")


if __name__ == "__main__":
    main()
