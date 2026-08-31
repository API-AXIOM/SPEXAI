#!/usr/bin/env bash
# GPU walker-batched MCMC counts-scan for the hot-element floor cross-check.
# One GPU process; walkers are batched into a single forward per emcee sub-step
# (emcee vectorized=True). Experiments (count levels) run SERIALLY. No CPU pool.
# sigma_v is now a free, per-walker parameter (the line deposition was
# vectorised); set SIGMA_V=<km/s> to pin it at that value for a reference run.
#
# GPU ACCELERATION (--tf32 --compile --fft32) IS ON BY DEFAULT AND SHOULD BE
# STANDARD FOR ALL PRODUCTION / REAL-SCIENCE INFERENCE RUNS: it is ~3x faster on
# the operator forward (A10) at an accuracy cost far below the emulator's own
# error -- TF32 tensor-core matmul ~1e-3, float32 continuum FFT ~1e-6, vs the
# emulator's ~3e-3 MRE (and the hot-floor study shows even 3e-3 does not bias
# science below ~1e7 counts). Set ACCEL="" only for a float64 reference run.
#
# The forward is now ELEMENT-BATCHED by default: all 30 trunks are evaluated as
# a few grouped GEMMs (vmap) which --compile then fuses, worth a further 2.19x
# over the old per-element serial loop at no accuracy cost (max rel vs fp64
# 6.24e-04 -> 6.22e-04). Pass --serial_forward to fall back to the loop.
#
# Prerequisites on the cluster (NO 40 GB caches needed -- truth is precomputed):
#   * repo importable as `spexai` (incl. the modified abundances.py /
#     operator_model.py and spexai/inference/data/tbabs_sigma.npz);
#   * spexai/models (the 30-element store) + response rsl_Hp_L_2025.rmf +
#     results/truth_single.npz REGENERATED against that store -- a truth built
#     from the old 28-element store has the same channel grid and so passes
#     every shape check while being silently wrong (dump_truth.py now records
#     the element set, and the loaders check it);
#   * conda env with torch(+CUDA), emcee, numpy, scipy.
#
# Usage (acceleration on by default; ~1-1.5 h per count level on an A10 at the
# defaults below; progress prints ~20 lines/level with a projected total time):
#   DEVICE=cuda COUNTS="4e4 1e6 1e8" ./scripts/experiments/hot_floor/run_cluster.sh single
set -euo pipefail

ENV=${SPEXAI_ENV:-spexai}
MODE=${1:-single}
DEVICE=${DEVICE:-cuda}
NWALKERS=${NWALKERS:-64}           # ensemble (>=2x ndim=11; 64 = one chunk/half)
NSTEPS=${NSTEPS:-800}              # ample for the ~11-D near-Gaussian posterior
CHUNK=${CHUNK:-32}                 # walker sub-batch (bounds GPU memory)
RESULTS=${SPEXAI_RESULTS:-$HOME/data/spexai_data/results}
TRUTH=${SPEXAI_TRUTH:-$RESULTS/hot_floor/truth_${MODE}.npz}
COUNTS=${COUNTS:-"4e4 1e6 1e8"}    # realistic / deep / near-N* (bias ~ 1 sigma)
# GPU acceleration -- STANDARD for production inference (see header); ACCEL=""
# disables it for a float64 reference run.
ACCEL=${ACCEL:-"--tf32 --compile --fft32"}

export MKL_THREADING_LAYER=GNU     # or torch import dies on the conda MKL stack
export SPEXAI_STORE=${SPEXAI_STORE:-$PWD/spexai/models}   # all 30 elements
export SPEXAI_RESPONSES=${SPEXAI_RESPONSES:-$HOME/work/data/spexai/responses}

echo "device=$DEVICE mode=$MODE walkers=$NWALKERS steps=$NSTEPS truth=$TRUTH"
echo "store=$SPEXAI_STORE responses=$SPEXAI_RESPONSES"
echo "accel: ${ACCEL:-<none: float64 reference>}  chunk=$CHUNK"
for c in $COUNTS; do
  echo "=== counts=$c ==="
  # --no-capture-output: stream stdout live (conda run buffers it otherwise,
  # which would leave the log empty until the buffer flushes)
  conda run --no-capture-output -n "$ENV" python -u scripts/experiments/hot_floor/mcmc_check.py \
    --vectorized --device "$DEVICE" --mode "$MODE" --counts "$c" \
    --truth_npz "$TRUTH" --nwalkers "$NWALKERS" --nsteps "$NSTEPS" \
    --chunk "$CHUNK" $ACCEL --tag "c${c}"
done
echo "done; results in $RESULTS/hot_floor/mcmc_${MODE}_vec_c*.npz"
