#!/usr/bin/env bash
# Build the project report (XeLaTeX + BibTeX, three passes) and print the page count.
set -e
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null || true
xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
grep -o "Output written on main.pdf ([0-9]* pages" main.log || true
grep -n "LaTeX Warning: \(Reference\|Citation\)" main.log || echo "no undefined references or citations"
