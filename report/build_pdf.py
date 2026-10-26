"""
build_pdf.py - turn report.md into report.pdf.

    pip install markdown                 (only needed for this script)
    python report/build_pdf.py           ->  report/report.pdf
    python report/build_pdf.py some.md   ->  some.pdf (next to the .md file)

How it works: the Markdown is converted to HTML with the `markdown` package, the
formulas ($...$ and $$...$$) are rendered by KaTeX, and Google Chrome (or Edge)
prints the page to PDF in headless mode. Image paths are relative to the .md file
(the figures come from ../results/, so run run_all.py first). Needs an internet
connection for the KaTeX script and the fonts.
"""

import html
import os
import re
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

CSS = """
@page { size: A4; margin: 14mm 16mm 14mm 16mm; }
html { font-size: 10pt; }
body { font-family: "Source Serif 4", Georgia, "Times New Roman", serif; line-height: 1.36;
       color: #1a1a1a; margin: 0; }
h1 { font-family: "Source Sans 3", "Segoe UI", Arial, sans-serif; font-size: 1.75rem;
     line-height: 1.15; margin: 0 0 0.3rem; }
h2 { font-family: "Source Sans 3", "Segoe UI", Arial, sans-serif; font-size: 1.22rem;
     margin: 1.1rem 0 0.4rem; padding-bottom: 0.15rem; border-bottom: 1px solid #cfd6cf;
     break-after: avoid; }
h3 { font-family: "Source Sans 3", "Segoe UI", Arial, sans-serif; font-size: 1.05rem; margin: 0.8rem 0 0.3rem; }
p, li { margin: 0.35rem 0; }
ul, ol { padding-left: 1.3rem; margin: 0.3rem 0; }
hr { border: 0; border-top: 1px solid #cfd6cf; margin: 0.8rem 0; }
code { font-family: "Source Code Pro", Consolas, monospace; font-size: 0.88em; background: #f1f3f0;
       padding: 0 0.2em; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; font-size: 0.92rem; break-inside: avoid; }
th, td { border: 1px solid #cfd6cf; padding: 0.25rem 0.45rem; vertical-align: top; text-align: left; }
th { background: #eef2ed; font-family: "Source Sans 3", "Segoe UI", Arial, sans-serif; }
blockquote { margin: 0.5rem 0; padding: 0.3rem 0.8rem; border-left: 3px solid #9fb3a5; background: #f5f7f4; }
p.figure { text-align: center; break-inside: avoid; margin: 0.6rem 0; }
p.figure img { max-width: 94%; display: block; margin: 0 auto 0.25rem; }
p.figure img.narrow { max-width: 52%; }
p.figure em { font-size: 0.9rem; color: #444; }
.katex-display { margin: 0.5rem 0; }
.appendix { break-before: page; }
"""


def protect_math(text):
    """Swap formulas for placeholders so markdown does not touch the underscores
    and asterisks inside them. Returns (text, list of (placeholder, html))."""
    found = []

    def keep(match, display):
        tex = match.group(1)
        key = f"MATHPLACEHOLDER{len(found)}X"
        wrapped = f"\\[{tex}\\]" if display else f"\\({tex}\\)"
        found.append((key, html.escape(wrapped, quote=False)))
        return key

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(m, True), text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", lambda m: keep(m, False), text)
    return text, found


def build_html(source, html_out):
    with open(source, encoding="utf-8") as f:
        text = f.read()
    text, formulas = protect_math(text)
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    for key, value in formulas:
        body = body.replace(key, value)

    # Paragraphs that hold an image get a centred figure style; Figure 0 is drawn smaller.
    body = re.sub(r"<p>(<img [^>]*>)", r'<p class="figure">\1', body)
    body = body.replace('alt="Figure 0"', 'alt="Figure 0" class="narrow"')
    # The learning record starts on a new page.
    body = body.replace("<h2>Appendix", '<h2 class="appendix">Appendix')

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CSCI3235 Midterm I report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Code+Pro&family=Source+Sans+3:wght@600;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<style>{CSS}</style>
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '\\\\[', right: '\\\\]', display: true}},
    {{left: '\\\\(', right: '\\\\)', display: false}}]}});"></script>
</head><body>
{body}
</body></html>"""
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(page)


def find_browser():
    for path in BROWSERS:
        if os.path.exists(path):
            return path
    sys.exit("Could not find Chrome or Edge. Open report.md in VS Code and use the "
             "'Markdown PDF' extension instead.")


def main():
    source = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "report.md")
    folder = os.path.dirname(source)
    # The temporary HTML sits next to the .md file so relative image paths still work.
    html_out = os.path.join(folder, "_report_print.html")
    pdf_out = os.path.splitext(source)[0] + ".pdf"

    build_html(source, html_out)
    browser = find_browser()
    url = "file:///" + html_out.replace("\\", "/")
    try:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=10000", f"--print-to-pdf={pdf_out}", url],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.remove(html_out)
    print("wrote", pdf_out)


if __name__ == "__main__":
    main()
