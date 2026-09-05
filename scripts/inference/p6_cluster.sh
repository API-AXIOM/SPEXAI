#!/bin/bash
# P6: the linearisation factor k = (real MLE bias) / (linearised b_sys), at one
# Tier B sweep point, on a GPU.
#
#     scripts/inference/p6_cluster.sh [POINT] [SEED_CHUNK]
#
# Needs an EXCLUSIVE card. Check before launching:
#     nvidia-smi --query-gpu=memory.used,memory.total --format=csv
#
# MEMORY: counts_torch(grad=True) does not chunk walkers, so the L-BFGS graph
# scales with the seeds in ONE call. 40 seeds in one call OOMed a 22 GB card at
# 20.84 GB; --seed_chunk 2 runs comfortably (measured 2026-09-03, and it
# matches the 2-4 row ceiling the batched-HMC work found for the same call).
# Rows are independent optimisations, so the split changes no result. Size a
# different card with p6_probe.py.
#
# PRECISION: SE(k) scales as 1/sqrt(K * N) and the Poisson likelihood costs the
# same at any N, so counts buy precision for free where seeds cost memory and
# wall-clock.
#
# CONVERGENCE: an under-converged fit sits at its truth start and reports k too
# SMALL -- the direction that falsely exonerates the linearisation. Read the
# "convergence:" line (drift as a fraction of the measured bias) before reading
# any k. FOUR flags below serve that, and they all address ONE root cause:
# every threshold in torch's LBFGS is ABSOLUTE, while these parameters span
# three decades in sigma (log_norm ~6e-5 vs sigma_v ~6e-2 at 1e9 counts), which
# is where ~1e6 of the measured cond(F) ~ 1e7 comes from.
#
#   --precondition  optimise in units of sigma_ref. THE fix; the rest are
#                   safety nets that stop mattering once this is on.
#   --tol_change 0  torch's 1e-9 is absolute, applied to the step size and to
#                   |loss - prev_loss|; real steps here sit at ~1e-9.
#   --max_eval      torch's default is max_iter*5//4 and caps FUNCTION EVALS,
#                   not iterations; step() also hands the line search
#                   max_eval-minus-evals-so-far as its own budget, so the late
#                   searches are starved. Raised here as insurance. NOTE it has
#                   never been observed to bind -- the "instant pass, 0.00e+00
#                   movement, -logL still descending" signature is the line
#                   search returning t = 0 exactly, whereupon d*t <=
#                   tolerance_change is 0 <= 0 and the pass breaks. Do NOT
#                   raise it much further: a big line-search budget plus a
#                   zeroed bracket guard is what produced NaN steps (see
#                   _line_search_hook in mle_reseed.py).
#   --ls_debug      prints iterations actually run, evals per line search, how
#                   many searches returned t=0, and max|grad| at entry. Without
#                   it a stalled optimiser and a converged one look identical.
set -e
export MKL_THREADING_LAYER=GNU          # or torch import dies on the cluster
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

# Mirrors spexai.config: $SPEXAI_RESULTS if set, else the cluster layout. Done
# in shell rather than by importing spexai, so resolving a path costs nothing
# and does not drag torch (or macOS's duplicate-libomp abort) into it.
RESULTS=${SPEXAI_RESULTS:-$HOME/data/spexai_data/results}
R=$RESULTS/bias_sweep

POINT=${1:-14}         # 14 = worst by |b_sys|/sigma_ref in the single-T sweep
CHUNK=${2:-2}
LEVELS=${3:-"1e7 1e8 1e9"}

# Inputs: the UNMASKED jsonl (its b_sys is what P6 calibrates -- the Fe XXV cut
# deletes the channels the emulator is worst at) and the STAMPED truth (the
# original predates rmf/arf recording and check_truth_response will refuse it,
# though its contents were verified identical).
BIAS=$R/bias_single_n20_s3.jsonl
TRUTH=$R/truth_single_n20_s3_stamped.npz

# Several count levels at the same point: k should be count-independent, since
# b_sys is and the MLE bias tends to a fixed pseudo-true offset. If they
# disagree, P7's correction has to be count-aware -- cheap to learn now, and
# Tier B quotes at 1e6, three decades below where k is easiest to measure.
#
# --deterministic: the line deposit's index_add_ has repeated indices, so CUDA
# sums in a varying order and identical parameters return slightly different
# likelihoods (~0.085 measured). Against the shallow likelihood well at 1e7
# that jitter stalls the line search; at 1e9 the well is ~60x deeper and it
# does not matter. Torch raises if it has no deterministic kernel for an op --
# that error is informative, not a failure of this script.
for N in $LEVELS; do
    echo "=== point $POINT, $N counts ==="
    python -u scripts/inference/mle_reseed.py \
        --method lbfgs --device cuda --deterministic \
        --bias_jsonl "$BIAS" --truth_npz "$TRUTH" \
        --point "$POINT" --counts $N --n_seeds 8 --seed_chunk "$CHUNK" \
        --max_iter 400 --n_restarts 3 --tol_change 0 \
        --precondition --max_eval 4000 --ls_debug \
        --out "$RESULTS/mle_reseed/p6_single_pt${POINT}_N${N}.npz"
done
