#!/usr/bin/env bash
# Compile the COSMOS-Web X-ray paper from LaTeX to PDF.
# Usage: ./compile_paper.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TEX_FILE="cosmos-web_xray_paper_standalone.tex"
BASE_NAME="${TEX_FILE%.tex}"

fallback_pdflatex() {
  echo "Running pdflatex/bibtex fallback..."
  pdflatex -interaction=nonstopmode "${TEX_FILE}"
  bibtex "${BASE_NAME}" || true
  pdflatex -interaction=nonstopmode "${TEX_FILE}"
  pdflatex -interaction=nonstopmode "${TEX_FILE}"
}

if command -v latexmk >/dev/null 2>&1; then
  echo "Running latexmk..."
  if ! latexmk -pdf -interaction=nonstopmode "${TEX_FILE}"; then
    echo "latexmk failed; falling back to manual pdflatex compilation."
    fallback_pdflatex
  fi
else
  echo "latexmk not found; falling back to pdflatex."
  fallback_pdflatex
fi

echo "Compilation finished. Output located at ${BASE_NAME}.pdf"
