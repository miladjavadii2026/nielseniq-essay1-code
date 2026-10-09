#!/usr/bin/env python3
"""Replace the single @@CONTENTS@@ marker with a static dot-leader Contents block (raw openxml).
Page numbers are PGnn tokens that resolve_toc.py fills after a pass-1 render.
Usage: toc_inject.py SRC.md OUT.md"""
import re, sys
src, out = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
assert s.count("@@CONTENTS@@") == 1, "need exactly one @@CONTENTS@@ marker"
body = s.split("@@CONTENTS@@", 1)[1]
entries = [(1 if m.group(1) == "#" else 2, m.group(2).strip())
           for m in re.finditer(r"(?m)^(#|##) (.+)$", body)]
esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
toc = ('```{=openxml}\n<w:p><w:pPr><w:spacing w:after="120"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:pPr>'
       '<w:r><w:rPr><w:b/><w:sz w:val="28"/></w:rPr><w:t>Contents</w:t></w:r></w:p>')
for k, (lvl, t) in enumerate(entries):
    ind = '<w:ind w:left="360"/>' if lvl == 2 else ""
    toc += (f'<w:p><w:pPr>{ind}<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9350"/></w:tabs>'
            f'<w:spacing w:after="28"/></w:pPr><w:r><w:t xml:space="preserve">{esc(t)}</w:t></w:r>'
            f'<w:r><w:tab/></w:r><w:r><w:t>PG{k:02d}</w:t></w:r></w:p>')
toc += "\n```\n\n```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```"
open(out, "w", encoding="utf-8").write(s.replace("@@CONTENTS@@", toc))
print(f"TOC injected: {len(entries)} entries")
