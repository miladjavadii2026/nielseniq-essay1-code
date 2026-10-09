#!/usr/bin/env python3
"""Stop mid-word splitting in Word tables: content-aware gridCol widths (longest word per
column), tblLayout=fixed, 9-pt runs. Usage: fix_tables.py IN.docx OUT.docx [--width 9360]"""
import re, sys, subprocess, zipfile, os, tempfile, xml.dom.minidom
src, dst = sys.argv[1], os.path.abspath(sys.argv[2])
WIDTH = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 9360
CHAR, PAD, MINCOL, MAXTOK, SZ = 105, 260, 700, 18, 18

def cells(tbl):
    return [["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", c)) for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)]
            for r in re.findall(r"<w:tr\b.*?</w:tr>", tbl, re.S)]

def longest(vals):
    return max((min(len(w.strip(".,;:()[]")), MAXTOK) for v in vals for w in re.split(r"[\s/\-,;]+", v) if w.strip(".,;:()[]")), default=0)

def widths(grid):
    n = max((len(r) for r in grid), default=0)
    if not n: return []
    col = lambda i: [r[i] for r in grid if i < len(r)]
    mins = [max(MINCOL, longest(col(i)) * CHAR + PAD) for i in range(n)]
    if sum(mins) <= WIDTH:
        avg = [sum(map(len, col(i))) / max(1, len(col(i))) for i in range(n)]
        left, tot = WIDTH - sum(mins), sum(avg) or 1
        w = [int(mins[i] + left * avg[i] / tot) for i in range(n)]
    else:
        k = WIDTH / sum(mins); w = [max(int(m * k), 600) for m in mins]
    w[w.index(max(w))] += WIDTH - sum(w)
    return w

def fix(tbl):
    w = widths(cells(tbl))
    if not w: return tbl
    tbl = re.sub(r'<w:tblW w:type="pct" w:w="\d+"\s*/>', f'<w:tblW w:type="dxa" w:w="{sum(w)}"/>', tbl)
    if "<w:tblLayout" not in tbl: tbl = tbl.replace("</w:tblPr>", '<w:tblLayout w:type="fixed"/></w:tblPr>', 1)
    gcs = list(re.finditer(r'<w:gridCol w:w="\d+"\s*/>', tbl))
    if len(gcs) == len(w):
        parts, pos = [], 0
        for g, wi in zip(gcs, w):
            parts += [tbl[pos:g.start()], f'<w:gridCol w:w="{wi}"/>']; pos = g.end()
        tbl = "".join(parts) + tbl[pos:]
    def run(m):
        r = m.group(0)
        if "<w:rPr>" in r:
            r = re.sub(r'<w:sz w:val="\d+"\s*/>', f'<w:sz w:val="{SZ}"/>', r) if "<w:sz " in r else r.replace("<w:rPr>", f'<w:rPr><w:sz w:val="{SZ}"/>', 1)
            if "<w:szCs" not in r: r = r.replace(f'<w:sz w:val="{SZ}"/>', f'<w:sz w:val="{SZ}"/><w:szCs w:val="{SZ}"/>', 1)
        else:
            r = r.replace("<w:r>", f'<w:r><w:rPr><w:sz w:val="{SZ}"/><w:szCs w:val="{SZ}"/></w:rPr>', 1)
        return r
    return re.sub(r"<w:r>(?:(?!</w:r>).)*</w:r>", run, tbl, flags=re.S)

tmp = tempfile.mkdtemp(); zipfile.ZipFile(src).extractall(tmp)
p = os.path.join(tmp, "word/document.xml"); doc = open(p, encoding="utf-8").read()
tbls = list(re.finditer(r"<w:tbl>.*?</w:tbl>", doc, re.S))
out, pos = [], 0
for m in tbls: out += [doc[pos:m.start()], fix(m.group(0))]; pos = m.end()
doc = "".join(out) + doc[pos:]
xml.dom.minidom.parseString(doc)
open(p, "w", encoding="utf-8").write(doc)
if os.path.exists(dst): os.remove(dst)
subprocess.run(f"cd {tmp} && zip -Xqr {dst} .", shell=True, check=True)
print(f"fixed {len(tbls)} tables -> {dst}")
