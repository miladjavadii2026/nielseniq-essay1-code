#!/usr/bin/env python3
"""Mechanical QA sweep. Usage: verify_report.py DOC.docx DOC.pdf [--registry numbers.txt] [--banned extra.txt]
Hard fails (exit 1): dashes in XML, PG tokens, blank pages, caption gaps/duplicates, registry numbers missing.
Reported: mid-word table breaks, banned terms with context."""
import re, sys, subprocess, zipfile
from collections import Counter
docx, pdf = sys.argv[1], sys.argv[2]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
pages = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout.split("\f")
if pages and not pages[-1].strip(): pages = pages[:-1]
txt = "\n".join(pages); fails = []
em, en = xml.count("\u2014"), xml.count("\u2013")
if em or en: fails.append(f"dashes em={em} en={en}")
if re.search(r"PG\d\d", txt): fails.append("unresolved PG tokens")
blank = [i + 1 for i, p in enumerate(pages) if len(p.strip()) < 5]
if blank: fails.append(f"blank pages {blank}")
lines = txt.split("\n")
breaks = [(lines[i].strip(), lines[i + 1].strip()) for i in range(len(lines) - 1)
          if 2 <= len(lines[i].strip()) <= 6 and lines[i].strip().isalpha() and lines[i + 1].strip()
          and lines[i + 1].strip()[0].islower() and len(lines[i + 1].strip()) <= 4 and lines[i + 1].strip().isalpha()]
body = txt.split("Contents", 1)[-1]
for kind in ("Figure", "Table"):
    caps = re.findall(rf"(?m)^{kind} ([A-D]?)(\d+)[.:]\s*(.*)$", body)
    series = {}
    for pre, num, title in caps: series.setdefault(pre, []).append(int(num))
    for pre, nums in series.items():
        if nums != list(range(1, len(nums) + 1)): fails.append(f"{kind} {pre or ''} sequence {nums}")
    dup = [c for c, n in Counter(f"{p}{n}" for p, n, _ in caps).items() if n > 1]
    if dup: fails.append(f"duplicate {kind} labels {dup}")
banned = ["advisor", "workstation", "sandbox", "scorecard", "FINAL_RERUN", "FINAL_VALIDATION", r"v11[a-z]?\b",
          "the the", "validation validation", "records records", "summary summarizes", r"\bPASS\b", r"\bFAIL\b",
          r"\b(Task|Run|Step|Stage) 0?\d+[A-B]?\b", "essay"]
if arg("--banned"): banned += [l.strip() for l in open(arg("--banned")) if l.strip()]
hits = {b: [txt[max(0, m.start() - 35):m.end() + 35].replace("\n", " ") for m in re.finditer(b, txt, re.I)] for b in banned}
hits = {b: v for b, v in hits.items() if v}
if arg("--registry"):
    miss = [n for n in (l.strip() for l in open(arg("--registry"))) if n and n not in txt]
    if miss: fails.append(f"registry numbers missing {miss}")
print(f"pages={len(pages)} em/en={em}/{en} blank={blank or 'none'} word-breaks={len(breaks)}")
for b in breaks[:10]: print("  break:", b)
for b, v in hits.items(): print(f"  term {b!r}: {len(v)}  e.g. {v[0]!r}")
print("HARD FAILS:", fails or "none")
sys.exit(1 if fails else 0)
