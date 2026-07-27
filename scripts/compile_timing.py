"""Compare steady-state training throughput across runs (e.g. --compile 0 vs 1).

torch.compile pays a large one-time graph-compilation cost, so a short run is
mostly warmup and hides the per-step benefit. This reads the per-eval
`elapsed_s` recorded in each run's `<tag>_history.json` and reports the
STEADY-STATE seconds/step, measured between two late evals so the first
interval (which absorbs the compile warmup) is excluded.

    python scripts/compile_timing.py \
        compiled=~/data/.../timing_compile/sweep_history.json \
        uncompiled=~/data/.../timing_nocompile/sweep_history.json
"""

import json
import sys


def steady_state_s_per_step(hist, warmup_evals=1):
    """Seconds/step between the (warmup_evals)-th eval and the last, so the
    initial interval (compile warmup, cache fill) is excluded."""
    h = hist["history"]
    if len(h) < warmup_evals + 2:
        return None, "too few evals"
    a, b = h[warmup_evals], h[-1]
    dstep = b["step"] - a["step"]
    dt = b["elapsed_s"] - a["elapsed_s"]
    if dstep <= 0:
        return None, "no step progress"
    return dt / dstep, (f"steps {a['step']}->{b['step']}, "
                        f"{dt:.0f}s over {dstep} steps")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: label=path/to/<tag>_history.json ...")
    rows = []
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        with open(path.rstrip()) as f:
            hist = json.load(f)
        sps, note = steady_state_s_per_step(hist)
        rows.append((label, sps, note))
        if sps is not None:
            print(f"{label:>12}: {sps * 1000:.1f} ms/step  ({note})")
        else:
            print(f"{label:>12}: n/a ({note})")
    valid = [(l, s) for l, s, _ in rows if s is not None]
    if len(valid) == 2:
        (l0, s0), (l1, s1) = valid
        # report the speedup of the faster over the slower
        fast, slow = (l1, l0) if s1 < s0 else (l0, l1)
        print(f"\n{fast} is {max(s0, s1) / min(s0, s1):.2f}x faster than {slow}")


if __name__ == "__main__":
    main()
