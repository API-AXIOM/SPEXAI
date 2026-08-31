"""Diagnostic comparison of trained elements: config, test metrics, and the
end-of-run val-MRE trajectory (to spot premature early-stop / noise).

    python scripts/compare_elements.py 5 11 13 16
"""
import argparse
import glob
import json
import os

NBINS = 24212
SYM = {5: "B", 11: "Na", 13: "Al", 16: "S", 4: "Be", 3: "Li"}


def find(runroot, z, name):
    c = glob.glob(os.path.join(runroot, f"element{z}", "tier1", name))
    if not c:
        c = glob.glob(os.path.join(runroot, f"element{z}", "**", name),
                      recursive=True)
    return c[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elements", nargs="+", type=int)
    ap.add_argument("--runroot", default=os.path.expanduser(
        "~/work/data/spexai/runs"))
    args = ap.parse_args()

    for z in args.elements:
        hp = find(args.runroot, z, "reweight_full_history.json")
        bp = find(args.runroot, z, "benchmark_test.json")
        if hp is None:
            print(f"Z{z}: no history"); continue
        h = json.load(open(hp)); a = h["args"]; hist = h["history"]
        decay = int((1 - (a.get("wsd_decay_frac") or 0.15)) * a["steps"])
        best, final = h["best"].get("step"), hist[-1]["step"]
        b = next(iter(json.load(open(bp)).values())) if bp else {}
        o = b.get("overall", {}); f = b.get("floor", {})
        nsp = o.get("n_spectra", 1)
        sig = 100 * (NBINS - f.get("n_floor_bins", 0) / max(nsp, 1)) / NBINS

        print(f"=== Z{z} {SYM.get(z, '?')} ===")
        print(f"  config: lr={a.get('lr')} signal_frac={a.get('signal_frac')} "
              f"mre_smooth={a.get('mre_smooth')} patience={a.get('early_stop_patience')} "
              f"hidden={a.get('hidden')} linehead={a.get('use_linehead')}")
        print(f"  signal bins: {sig:.2f}% of grid")
        if o:
            print(f"  TEST: MRE={o['mre_mean']:.4f} med={o['mre_median']:.4f} "
                  f"y@1%={o['yield_1pct']:.1f} y@0.1%={o['yield_01pct']:.1f} "
                  f"floorviol={f.get('violation_pct', float('nan')):.3f}%")
        print(f"  steps: best={best} final={final} decay_starts={decay} "
              f"({'STOPPED PRE-DECAY -> ' + ('still-improving?' if best == final else 'plateaued') if final < decay else 'ran full'})")
        # end-of-run val-MRE trend: last 5 evals, % change
        prev = None
        tail = []
        for r in hist[-6:]:
            ch = "" if prev is None else f"{(prev-r['val_mre_mean'])/prev*100:+.0f}%"
            tail.append(f"{r['step']}:{r['val_mre_mean']:.4f}({ch})")
            prev = r["val_mre_mean"]
        print("  tail:", "  ".join(tail))
        print()


if __name__ == "__main__":
    main()
