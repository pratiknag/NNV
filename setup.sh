#!/usr/bin/env bash
# Installs everything needed by run_simulations.sh and run_application.sh. Run from the
# repository root:
#
#   bash setup.sh
#
#   PYTHON_BIN   Python interpreter used to create env/ (default python3.11, else python3)
#   SKIP_R=1     skip the R packages
#   SKIP_PY=1    skip the Python environment
#
# Python: a virtual environment env/ with the packages of requirements.txt (PyTorch from the
# CPU wheel index; no GPU is needed). R: GPvecchia, fields and Matrix, installed into the
# user library if the system library is not writable.
# Also needed, not installed here: a C/C++ compiler and git (for veccs and the R packages),
# and Arial or a metric-compatible font such as Liberation Sans (for patterns.R).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

step() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
if [[ "${SKIP_PY:-0}" != "1" ]]; then
  if [[ -n "${PYTHON_BIN:-}" ]]; then PY=$PYTHON_BIN
  elif command -v python3.11 > /dev/null; then PY=python3.11
  else PY=python3; fi
  "$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 is required"'
  if [[ ! -x env/bin/python ]]; then
    step "creating env/ with $($PY --version)"
    "$PY" -m venv env
  fi
  step "installing the Python packages"
  env/bin/python -m pip install --upgrade pip wheel
  env/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
  env/bin/python -m pip install -r requirements.txt
  env/bin/python -c "import numpy, scipy, pandas, matplotlib, seaborn, numba, torch, veccs; print('Python packages OK')"
fi

# ---------------------------------------------------------------------------
# R
# ---------------------------------------------------------------------------
if [[ "${SKIP_R:-0}" != "1" ]]; then
  command -v Rscript > /dev/null || { echo "Rscript not found: install R (>= 4.1) first"; exit 1; }
  step "installing the R packages"
  Rscript - <<'RSCRIPT'
lib <- .libPaths()[1]
if (file.access(lib, 2) != 0) {
  lib <- Sys.getenv("R_LIBS_USER")
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
repos <- "https://cloud.r-project.org"
for (p in c("Matrix", "fields", "remotes")) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib, repos = repos)
}
if (!requireNamespace("GPvecchia", quietly = TRUE)) {
  ok <- tryCatch({ install.packages("GPvecchia", lib = lib, repos = repos)
                   requireNamespace("GPvecchia", quietly = TRUE) },
                 error = function(e) FALSE)
  # GPvecchia is not always available on CRAN: fall back to GitHub
  if (!isTRUE(ok)) remotes::install_github("katzfuss-group/GPvecchia", lib = lib)
}
for (p in c("GPvecchia", "fields", "Matrix")) {
  cat(sprintf("%-10s %s\n", p, as.character(packageVersion(p))))
}
cat("R packages OK (tested with R 4.3.3, GPvecchia 0.1.8, fields 16.3.1, Matrix 1.6.5)\n")
RSCRIPT
fi
step "setup done"
