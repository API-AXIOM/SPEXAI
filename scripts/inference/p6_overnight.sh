#!/usr/bin/env bash
#
# Overnight P6: truth -> bias -> Gauss-Newton -> analysis, for N sweep points.
#
#   nohup bash scripts/inference/p6_overnight.sh > logs/p6_overnight.log 2>&1 &
#
# Every stage is resumable and skips finished work, so re-running the same
# command after a crash, a walltime kill, or a short night continues where it
# stopped. That is the design point: at measured costs 100 points is ~15 h, so
# a partial result has to be a usable result.
#
# Stage costs (per point): truth ~18 s (CPU, reads the 40 GB SPEX caches --
# stage_truth hardcodes device="cpu", so it cannot be GPU'd and does not need
# to be), bias ~262 s (LAPTOP CPU figure, never timed on GPU), GN ~272 s
# (measured, 20-point sweep 2026-09-06).
#
# Override anything from the environment:
#   NPOINTS=50 SEED=7 bash scripts/inference/p6_overnight.sh
#
set -euo pipefail

NPOINTS="${NPOINTS:-100}"
SEED="${SEED:-3}"
MODE="${MODE:-single}"
COUNTS="${COUNTS:-1e9}"
NSEEDS="${NSEEDS:-8}"          # independent STARTS in noiseless mode
GN_ITER="${GN_ITER:-12}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Mirrors spexai.config.RESULTS exactly, so overriding SPEXAI_RESULTS moves the
# scripts' outputs and this driver's paths together instead of silently apart.
RESULTS_DIR="${SPEXAI_RESULTS:-$HOME/data/spexai_data/results}"
SWEEP_DIR="${SWEEP_DIR:-$RESULTS_DIR/bias_sweep}"
MLE_DIR="${MLE_DIR:-$RESULTS_DIR/mle_reseed}"
LOGDIR="${LOGDIR:-$REPO/logs}"

TAG="${MODE}_n${NPOINTS}_s${SEED}"
TRUTH_NPZ="$SWEEP_DIR/truth_${TAG}.npz"
BIAS_JSONL="$SWEEP_DIR/bias_${TAG}.jsonl"
P6_JSONL="$MLE_DIR/p6_gn_${TAG}.jsonl"

# Without this torch import dies on the cluster (conda-MKL numpy vs libgomp).
export MKL_THREADING_LAYER=GNU
# p6_sweep sets this itself under --deterministic, but the bias stage does not.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOGDIR" "$SWEEP_DIR" "$MLE_DIR"

say() { printf '\n[%s] === %s ===\n' "$(date '+%F %T')" "$*"; }

# Run a stage, time it, and stop the pipeline if it fails. Stages are chained
# with && rather than run unconditionally: a failed truth stage would otherwise
# be followed by a bias stage reading a half-written npz.
run_stage() {
    local name="$1"; shift
    local log="$LOGDIR/p6_overnight_${name}_${TAG}.log"
    say "STAGE $name -> $log"
    local t0 rc=0
    t0=$(date +%s)
    # Capture the status explicitly: inside `if ! cmd; then`, $? is the
    # NEGATED status (0), so the failure message would report a clean exit.
    "$@" >> "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        printf '[%s] STAGE %s FAILED (exit %d). Last 25 lines:\n' \
            "$(date '+%F %T')" "$name" "$rc"
        tail -25 "$log"
        printf 'Stages are resumable: fix the cause and re-run the same '
        printf 'command to continue.\n'
        exit 1
    fi
    printf '[%s] stage %s done in %d s\n' \
        "$(date '+%F %T')" "$name" "$(( $(date +%s) - t0 ))"
}

say "P6 overnight: $NPOINTS points, tag $TAG, $COUNTS counts, $NSEEDS starts"
printf 'repo       %s\ntruth npz  %s\nbias jsonl %s\np6 jsonl   %s\n' \
    "$REPO" "$TRUTH_NPZ" "$BIAS_JSONL" "$P6_JSONL"
printf '\nNOTE: an LHS of %d points is a DIFFERENT point set from the existing\n' \
    "$NPOINTS"
printf 'n20_s3 sweep (qmc.LatinHypercube(...).random(n) does not nest), so none\n'
printf 'of the existing truth/bias work carries over and the two sets are\n'
printf 'separate samples of the same ranges -- poolable for statistics, but not\n'
printf 'one design.\n'

if [ "$SKIP_PREFLIGHT" != "1" ]; then
    say "PREFLIGHT"
    python -u "$REPO/scripts/inference/p6_preflight.py" \
        --n_points "$NPOINTS" --sweep_dir "$SWEEP_DIR"
fi

# --- 1. SPEX truth spectra. CPU + caches; resumable per ELEMENT (~30 of them),
#        not per point, so a kill loses at most one element's pass.
run_stage truth python -u "$REPO/scripts/inference/bias_sweep.py" \
    --stage truth --mode "$MODE" --n_points "$NPOINTS" --seed "$SEED" \
    --out "$SWEEP_DIR" --resume

# --- 2. Jacobian + Fisher -> b_sys. GPU, resumable per point.
run_stage bias python -u "$REPO/scripts/inference/bias_sweep.py" \
    --stage bias --mode "$MODE" --n_points "$NPOINTS" --seed "$SEED" \
    --device cuda --out "$SWEEP_DIR" --resume

# --- 3. P6: does the real fit move by b_sys? Gauss-Newton, GPU, resumable per
#        point. Explicit --truth_npz because p6_sweep defaults to the
#        *_stamped* copy of the n20 run, which does not exist for a fresh tag
#        (the preflight checks that the stamp is metadata, not a transform).
run_stage gn python -u "$REPO/scripts/inference/p6_sweep.py" \
    --device cuda --deterministic --noiseless \
    --counts "$COUNTS" --n_seeds "$NSEEDS" --seed_chunk "$NSEEDS" \
    --method gn --gn_iter "$GN_ITER" \
    --bias_jsonl "$BIAS_JSONL" --truth_npz "$TRUTH_NPZ" \
    --out "$P6_JSONL" --resume

# --- 4. Analysis. Cheap, and running it here means the morning starts with
#        numbers rather than a jsonl. Never fatal: the science data is already
#        on disk by this point, so a plotting-level bug must not look like a
#        failed run.
say "ANALYSIS"
python -u "$REPO/scripts/inference/p6_check_outliers.py" "$P6_JSONL" \
    || echo "check_outliers failed; the jsonl is intact, rerun by hand"
python -u "$REPO/scripts/inference/p6_trends.py" "$P6_JSONL" \
    || echo "trends failed; the jsonl is intact, rerun by hand"

say "DONE"
printf 'results: %s\n' "$P6_JSONL"
printf 'To pool with the existing 20-point sweep, pass both jsonls to the\n'
printf 'analysis scripts -- they are independent samples of the same ranges.\n'
