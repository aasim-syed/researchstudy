#!/usr/bin/env python3
"""Render docs/technical_writeup.md to docs/technical_writeup.pdf (markdown + weasyprint)."""
import os

import markdown
from weasyprint import HTML

HERE = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(HERE, "technical_writeup.md")).read()
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 16mm 15mm; @bottom-right { content: counter(page); font-size: 8pt; color: #777; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.2pt; line-height: 1.38; color: #1f2328; }
h1 { font-size: 17pt; margin: 0 0 4pt; } h2 { font-size: 12.5pt; border-bottom: 1px solid #d0d7de; margin-top: 14pt; }
h3 { font-size: 10.5pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8pt; width: 100%; }
td, th { border: 1px solid #d0d7de; padding: 2pt 4pt; vertical-align: top; } th { background: #f6f8fa; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #f6f8fa; padding: 0 2pt; }
pre { background: #f6f8fa; padding: 6pt; font-size: 7.6pt; white-space: pre-wrap; }
img { max-width: 100%; margin: 4pt 0; page-break-inside: avoid; }
p img { display: block; }
"""
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
HTML(string=html, base_url=HERE).write_pdf(os.path.join(HERE, "technical_writeup.pdf"))
print("wrote", os.path.join(HERE, "technical_writeup.pdf"))
