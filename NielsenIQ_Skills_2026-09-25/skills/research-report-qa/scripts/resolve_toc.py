#!/usr/bin/env python3
"""Two-pass TOC page resolution. Usage: resolve_toc.py PASS1.docx PASS1.pdf OUT.docx
Body starts after the LAST page still showing a PG token; dotted TOC lines are skipped."""
import re, sys, subprocess, zipfile, os, tempfile
docx, pdf, out = sys.argv[1:4]
z = zipfile.ZipFile(docx); xml = z.read("word/document.xml").decode("utf-8")
entries = re.findall(r'<w:t xml:space="preserve">([^<]+)</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>(PG\d\d)</w:t>', xml)
pages = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout.split("\f")
toc_pages = [i for i, p in enumerate(pages) if re.search(r"PG\d\d", p)]
body = (max(toc_pages) + 1) if toc_pages else 0
unesc = lambda t: t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
mp, missing = {}, []
for title, tok in entries:
    needle = unesc(title)[:30]
    rng = reversed(range(body, len(pages))) if title.strip().lower() == "references" else range(body, len(pages))
    found = next((i + 1 for i in rng
                  if any(needle in ln and "....." not in ln for ln in pages[i].split("\n"))), None)
    if found is None: missing.append(title)
    mp[tok] = found or 1
for tok, pg in mp.items(): xml = xml.replace(f"<w:t>{tok}</w:t>", f"<w:t>{pg}</w:t>")
tmp = tempfile.mkdtemp(); z.extractall(tmp)
open(os.path.join(tmp, "word/document.xml"), "w", encoding="utf-8").write(xml)
out = os.path.abspath(out)
if os.path.exists(out): os.remove(out)
subprocess.run(f"cd {tmp} && zip -Xqr {out} .", shell=True, check=True)
left = len(re.findall(r"PG\d\d", xml))
print(f"resolved {len(entries) - len(missing)}/{len(entries)} | unresolved: {missing or 'none'} | tokens left: {left}")
sys.exit(1 if missing else 0)
