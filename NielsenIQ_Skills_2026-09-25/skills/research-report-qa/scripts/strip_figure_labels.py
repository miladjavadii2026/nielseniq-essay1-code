#!/usr/bin/env python3
"""Mask workflow labels burned into figure PNGs (Task 03:, Step 6, Stage 11, Run4, PASS, sandbox).
Tesseract word boxes; masks the label plus a following number/colon/v3/dash token with the local
background; optional text replacement; re-OCRs and reports residuals.
Usage: strip_figure_labels.py IN.png OUT.png [--replace 'sandbox)=sample)' ...] [--pattern REGEX]"""
import re, sys, pytesseract
from PIL import Image, ImageDraw, ImageFont
src, dst = sys.argv[1], sys.argv[2]
pat = re.compile(sys.argv[sys.argv.index("--pattern") + 1] if "--pattern" in sys.argv else
                 r"^(Task|Step|Stage|Run|Tasks)\d*[:.]?$|^(FINAL_RERUN|PROVISIONAL_PASS|NOT_RUN_WORKSTATION|PASS|Stage\d+|Run\d+)", re.I)
repl = dict(a.split("=", 1) for i, a in enumerate(sys.argv) if i > 0 and sys.argv[i - 1] == "--replace")
im = Image.open(src).convert("RGB"); d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
W = [(d["left"][i], d["top"][i], d["width"][i], d["height"][i], d["text"][i]) for i in range(len(d["text"])) if d["text"][i].strip()]
draw = ImageDraw.Draw(im); font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def bg(x, y, w, h):
    strip = im.crop((max(0, x - 2), max(0, y - 9), min(im.width, x + w + 2), max(1, y - 6))).getcolors(10 ** 6)
    return max(strip, key=lambda c: c[0])[1] if strip else (255, 255, 255)
def mask(x, y, w, h, pad=3): draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=bg(x, y, w, h))
n = 0
for i, (x, y, w, h, t) in enumerate(W):
    if t in repl:
        mask(x, y, w, h); draw.text((x, y - 1), repl[t], fill=(30, 30, 30), font=ImageFont.truetype(font, max(10, int(h * .9)))); n += 1
    elif pat.search(t):
        x2 = x + w
        for j in (i + 1, i + 2):
            if j < len(W) and abs(W[j][1] - y) < 12 and W[j][0] - x2 < 40 and re.match(r"^(\d+[:.]?|v\d|[\u2014\u2013-]+)$", W[j][4]):
                x2 = W[j][0] + W[j][2]
        mask(x, y, x2 - x, h); n += 1
im.save(dst)
res = [t for t in pytesseract.image_to_string(Image.open(dst)).split() if pat.search(t) or t in repl]
print(f"masked {n} label(s) -> {dst}; residual after re-OCR: {res or 'none'}")
