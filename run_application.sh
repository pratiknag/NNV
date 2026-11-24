#!/usr/bin/env bash
# Application to the MIRS total precipitable water (TPW) data: Section 4 of the manuscript
# (Table 1, Figures 7-8) and Section S7 of the supplement (Figures S7-S8). Runs NNV, HV,
# low-rank (LR) and the exact Kalman filter on the 247 x 247 grid and writes the table and
# figures. Run from the repository root:
#
#   setsid nohup bash run_application.sh > logs/run_application.log 2>&1 < /dev/null &
#
#   PYTHON     Python interpreter (default env/bin/python if present, else python3)
#   FORCE=1    rerun every stage. By default a computing stage is skipped when its output
#              exists, so an interrupted run can be resumed. The table and figures are
#              always redrawn.
#   DRY_RUN=1  print the commands without running them
#   SKIP_EXACT=1  skip the exact Kalman filter (needs about 60 GB of RAM); Table 1 and
#              Figures 7 and S8 need it, Figures 8 and S7 do not
#
# Data: data/MIRS.csv.gz (the MIRS data of Jurek and Katzfuss, 2022,
# https://github.com/katzfuss-group/vecchiaFilter, data-application/MIRS.csv). The inner
# 247 x 247 cells (n = 61,009) of the western 263 x 263 grid, days 16-26, random 10% of the
# observations at each time point held out.
# Model parameters (Section 4): exponential Sigma_{1|0} and Q_t with variances 54 and 10 and
# range 0.2, noise variance tau^2 = 0.0697, diffusion 4.37e-8, advection 1.60e-4.
# Sparsity matched at about 300 nonzeros per row of L: NNV rho = 5 (Stage 2 on the pattern of
# U U^T), HV requested m = 393, LR N = 307.
#
# Outputs:
#   application_mirs/            prepared data (data.npz, grid.csv) and HV/LR patterns
#   application_mirs/results/    scores and filtering fields of every method, table_exact.tex
#   figures/application-*.pdf    Figures 7, 8, S7, S8
# Per-stage logs are written to logs/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""

if [[ -n "${PYTHON:-}" ]]; then P=$PYTHON
elif [[ -x env/bin/python ]]; then P=env/bin/python
else P=python3; fi
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXACT=${SKIP_EXACT:-0}
PY=src/python
R=src/R
APP=application_mirs
OUT=$APP/results
# var0, varQ, range, smoothness, tau^2, diffusion, advection
THETA="54.0 10.0 0.2 0.5 0.0696949381771818 4.366135371198055e-08 0.0001599539794219"
RHO=5
HV_M=393
LR_N=307
mkdir -p logs figures

step() { echo "[$(date '+%F %T')] $*"; }
run()  { local log=$1; shift; if [[ "$DRY_RUN" == "1" ]]; then echo "  $* > $log"; else "$@" > "$log" 2>&1; fi; }
todo() { [[ "$FORCE" == "1" || ! -e $1 ]]; }

# 1. Data: 247 x 247 grid, 11 days, random 10% hold-out
if todo $APP/data.npz; then
  step "preparing the MIRS data"
  run logs/app_prepare.log $P $PY/mirs_prepare.py --input=data/MIRS.csv.gz --output-dir=$APP
fi

# 2. HV and LR orderings and conditioning sets from GPvecchia
if todo $APP/patterns/lr_N$LR_N.nn; then
  step "GPvecchia patterns: HV m = $HV_M, LR N = $LR_N"
  run logs/app_patterns.log Rscript $R/export_patterns_sweep.R --side=247 \
    --conditioning_sizes=$HV_M --lr_sizes=$LR_N --output_dir=$APP/patterns
fi

# 3. NNV, HV and LR filters (timed one after the other), with the held-out scores and the
#    filtering means and standard deviations
mkdir -p $OUT
for METHOD in NNV HV LR; do
  if todo $OUT/fields_random_$METHOD.npz; then
    step "filter: $METHOD"
    run logs/app_filter_$METHOD.log $P $PY/mirs_validate7.py --theta $THETA --rho $RHO \
      --schemes random --methods $METHOD --hv-size $HV_M --lr-size $LR_N \
      --save-fields --out $OUT
  fi
done

# 4. Exact Kalman filter (dense, about 60 GB of RAM)
if [[ "$SKIP_EXACT" != "1" ]] && todo $OUT/fields_random_exact.npz; then
  step "exact Kalman filter"
  run logs/app_exact.log $P $PY/mirs_exact_full.py --data-dir=$APP --out=$OUT
fi

# 5. Table and figures
step "Figures 8 and S7"
run logs/app_maps.log $P $PY/mirs_maps.py
if [[ -e $OUT/fields_random_exact.npz || "$DRY_RUN" == "1" ]]; then
  step "Table 1, Figures 7 and S8"
  run logs/app_exact_plot.log $P $PY/mirs_exact_plot.py
fi

step "APPLICATION DONE"
