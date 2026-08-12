#!/usr/bin/env bash
# GPU walker-batched MCMC counts-scan for the hot-element floor cross-check.
# One GPU process; walkers are batched into a single forward per emcee sub-step
# (emcee vectorized=True). Experiments (count levels) run SERIALLY. No CPU pool.
# Tier 1: sigma_v is fixed at truth (freeing it needs the per-walker line-
# broadening vectorisation -- the required next step).
#
# Prerequisites on the cluster (NO 40 GB caches needed -- truth is precomputed):
#   * repo importable as `spexai` (incl. the modified abundances.py /
#     operator_model.py and spexai/inference/data/tbabs_sigma.npz);
#   * store28 + response rsl_Hp_L_2025.rmf + results/truth_single.npz;
#   * conda env with torch(+CUDA), emcee, numpy, scipy.
#
# Usage:
#   DEVICE=cuda NWALKERS=200 NSTEPS=3000 COUNTS="4e4 1e6 1e8" \
#       ./inference_demo/hot_floor/run_cluster.sh single
set -euo pipefail

ENV=${SPEXAI_ENV:-spexai}
MODE=${1:-single}
DEVICE=${DEVICE:-cuda}
NWALKERS=${NWALKERS:-200}          # big batch -> saturates the GPU
NSTEPS=${NSTEPS:-3000}
TRUTH=${SPEXAI_TRUTH:-inference_demo/hot_floor/results/truth_${MODE}.npz}
COUNTS=${COUNTS:-"4e4 1e6 1e8"}    # realistic / deep / near-N* (bias ~ 1 sigma)

export MKL_THREADING_LAYER=GNU     # or torch import dies on the conda MKL stack
export SPEXAI_STORE=${SPEXAI_STORE:-$PWD/inference_demo/hot_floor/store28}
export SPEXAI_RESPONSES=${SPEXAI_RESPONSES:-$HOME/work/data/spexai/responses}

echo "device=$DEVICE mode=$MODE walkers=$NWALKERS steps=$NSTEPS truth=$TRUTH"
echo "store=$SPEXAI_STORE responses=$SPEXAI_RESPONSES"
for c in $COUNTS; do
  echo "=== counts=$c ==="
  conda run -n "$ENV" python -u inference_demo/hot_floor/mcmc_check.py \
    --vectorized --device "$DEVICE" --mode "$MODE" --counts "$c" \
    --truth_npz "$TRUTH" --nwalkers "$NWALKERS" --nsteps "$NSTEPS" --tag "c${c}"
done
echo "done; results in inference_demo/hot_floor/results/mcmc_${MODE}_vec_c*.npz"
