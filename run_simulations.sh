#!/usr/bin/env bash
# Methodology figures and simulation study: Figures 2-6 of the manuscript and Figures S1-S6 of
# the supplement. Generates the data, runs NNV, HV, low-rank (LR) and the exact posterior or
# Kalman filter, and writes every figure. Run from the repository root:
#
#   setsid nohup bash run_simulations.sh > logs/run_simulations.log 2>&1 < /dev/null &
#
#   PYTHON         Python interpreter (default env/bin/python if present, else python3)
#   FORCE=1        regenerate the data and rerun every stage. By default a data or computing
#                  stage is skipped when its output exists, so an interrupted run can be
#                  resumed. Figures are always redrawn.
#   DRY_RUN=1      print the commands without running them
#   FILTER_MODELS  data models of the large-n filtering run (default gaussian,poisson,gamma,
#                  the models of Figure S6)
#
# Settings (Section 3): exponential covariance with variance 1 and range 0.15 on regular grids
# of [0, 1]^2; Gaussian (noise variance 0.2, filtering 0.25), logistic, Poisson and gamma data.
# All methods are matched on the average number of nonzeros per row of the posterior
# covariance factor L (NNV radius rho, HV requested conditioning-set size m, LR size N; the
# matched values were found with src/R/match_hv_nnz.R).
#
# Outputs (see README.md for the full list):
#   figures/      every figure of Sections 2-3 and S6
#   simulation_<model>/, filtering_<model>/   simulated data at n = 1,600
#   results_sparsity/   n = 1,600 results (spatial.csv, filter.csv)
#   large_n/            large-n results (spatial80_side*.csv, filter80_side300.csv,
#                       gpvecchia80_side*.csv), patterns and data
# Per-stage logs are written to logs/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""

if [[ -n "${PYTHON:-}" ]]; then P=$PYTHON
elif [[ -x env/bin/python ]]; then P=env/bin/python
else P=python3; fi
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
FILTER_MODELS=${FILTER_MODELS:-gaussian,poisson,gamma}
PY=src/python
R=src/R
MODELS="gaussian logistic poisson gamma"
mkdir -p logs figures

step() { echo "[$(date '+%F %T')] $*"; }
# run LOG CMD...: run CMD with its output in LOG
run()  { local log=$1; shift; if [[ "$DRY_RUN" == "1" ]]; then echo "  $* > $log"; else "$@" > "$log" 2>&1; fi; }
# todo FILE: true if FILE must be (re)made
todo() { [[ "$FORCE" == "1" || ! -e $1 ]]; }

# ---------------------------------------------------------------------------
# 1. Methodology figures (Section 2)
# ---------------------------------------------------------------------------
step "Figure 2: sparsity patterns"
run logs/patterns.log Rscript $R/patterns.R
step "Figure 3: implied prior basis functions of NNV and HV"
run logs/basis_functions.log $P $PY/basis_functions_comparison.py

# ---------------------------------------------------------------------------
# 2. Simulated data at n = 1,600 (40 x 40 grid), 10 data sets per data model
# ---------------------------------------------------------------------------
for MODEL in $MODELS; do
  if todo simulation_$MODEL/${MODEL}_seed10_domain1_dim2_n1600_smooth0.5_range0.15.csv; then
    step "spatial data: $MODEL"
    run logs/data_spatial_$MODEL.log Rscript $R/generate_Gaussian_datasets.R \
      --model=$MODEL --n=1600 --n_datasets=10 --output_dir=simulation_$MODEL
  fi
  if todo filtering_$MODEL/${MODEL}_seed10_n1600_T20_range0.15.csv; then
    step "filtering data: $MODEL (advection-diffusion, T = 20)"
    run logs/data_filtering_$MODEL.log Rscript $R/generate_advdiff_datasets.R \
      --model=$MODEL --n=1600 --n_times=20 --n_datasets=10 --output_dir=filtering_$MODEL \
      --noise_variance=0.25 --frac_obs=0.1 --diffusion=0.00004 --advection=0.01
  fi
done

# ---------------------------------------------------------------------------
# 3. Sections 3.2-3.3: NNV, HV and LR at five matched sparsity levels (n = 1,600)
# ---------------------------------------------------------------------------
# HV and LR orderings and conditioning sets from GPvecchia; levels 2-6 of LEVELS in
# src/python/sim_sparsity.py (about 40, 60, 80, 100 and 120 nonzeros per row)
if todo large_n/patterns/side40/lr_N122.nn; then
  step "GPvecchia patterns, 40 x 40 grid"
  run logs/patterns_side40.log Rscript $R/export_patterns_sweep.R --side=40 \
    --conditioning_sizes=58,87,124,150,181 --lr_sizes=41,60,82,99,122 \
    --output_dir=large_n/patterns/side40
fi
mkdir -p results_sparsity
for STUDY in spatial filter; do
  if todo results_sparsity/$STUDY.csv; then
    step "n = 1,600: $STUDY study (exact, NNV, HV, LR)"
    run logs/sparsity_$STUDY.log $P $PY/sim_sparsity.py --study=$STUDY --levels=2,3,4,5,6
  fi
done
step "Figures 4-5, S2-S3, S5"
run logs/plot_sparsity.log $P $PY/plot_sparsity.py

# ---------------------------------------------------------------------------
# 4. Boundary effects: exact, HV and NNV posterior means for one data set (Figures 1, S1)
# ---------------------------------------------------------------------------
for MODEL in logistic poisson; do
  if todo simulation_$MODEL/fields_seed1_n1600_hv_m.csv; then
    step "exact and HV posterior means: $MODEL"
    run logs/fields_$MODEL.log Rscript $R/export_spatial_fields.R --model=$MODEL --n=1600 \
      --dataset=1 --conditioning_sizes=10
  fi
done
step "Figures 1 and S1"
run logs/plot_spatial_maps.log $P $PY/plot_spatial_maps.py --models=logistic,poisson

# ---------------------------------------------------------------------------
# 5. Section 3.4: large n, all methods at about 80 nonzeros per row
# ---------------------------------------------------------------------------
# grid side, NNV rho, HV requested m, LR N
CONFIGS=("100 2.6 105 80" "200 2.23 102 77" "300 2.1 99 78")
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  if todo large_n/patterns/side$1/lr_N$4.nn; then
    step "GPvecchia patterns, $1 x $1 grid"
    run logs/patterns_side$1.log Rscript $R/export_patterns_sweep.R --side=$1 \
      --conditioning_sizes=$3 --lr_sizes=$4 --output_dir=large_n/patterns/side$1
  fi
done
# spatial-only, 10 data sets per data model (also writes the data to large_n/data/)
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  if todo large_n/spatial80_side$1.csv; then
    step "large n, spatial: n = $(( $1 * $1 ))"
    run logs/large_n_spatial_side$1.log $P $PY/large_n_spatial.py --side=$1 --rhos=$2 \
      --hv-sizes=$3 --lr-sizes=$4 --out-name=spatial80_side$1.csv
  fi
done
# filtering on the 300 x 300 grid, 10 data sets, T = 10
if todo large_n/filter80_side300.csv; then
  step "large n, filtering: n = 90000 ($FILTER_MODELS)"
  run logs/large_n_filter_side300.log $P $PY/large_n_filter.py --side=300 --n-datasets=10 \
    --rhos=2.1 --hv-sizes=99 --lr-sizes=78 --models=$FILTER_MODELS
  [[ "$DRY_RUN" == "1" ]] || mv large_n/filter_side300.csv large_n/filter80_side300.csv
fi
# HV and LR in GPvecchia on the same data (times in Figure 6)
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  if todo large_n/gpvecchia80_side$1.csv; then
    step "GPvecchia HV and LR: n = $(( $1 * $1 ))"
    run logs/gpvecchia_side$1.log Rscript $R/large_n_gpvecchia.R --side=$1 --n_datasets=10 \
      --conditioning_sizes=$3 --lr_sizes=$4 --output=large_n/gpvecchia80_side$1.csv
  fi
done
step "Figures 6, S4, S6"
run logs/plot_large_n80.log $P $PY/plot_large_n80.py

step "SIMULATIONS DONE"
