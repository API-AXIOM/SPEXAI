"""Pre-flight checks for the overnight P6 pipeline. Cheap; run before the 15 h.

This campaign's failures have all been silent and reassuring
(non-convergence read as k -> 0, a detectability rule that loosened as the
optimiser improved, drift that measured nothing). So the driver refuses to
start until the things that would waste a night are checked explicitly.

Exit code 0 = safe to run, 1 = stop.
"""
import argparse
import os
import sys

import numpy as np


def ok(msg):
    print(f"  OK    {msg}", flush=True)


def bad(msg):
    print(f"  FAIL  {msg}", flush=True)


def warn(msg):
    print(f"  WARN  {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_points", type=int, default=100)
    ap.add_argument("--store", default=os.environ.get("SPEXAI_STORE", ""))
    ap.add_argument("--sweep_dir", required=True,
                    help="bias_sweep results dir (holds truth_*/bias_*)")
    ap.add_argument("--ref_tag", default="single_n20_s3",
                    help="existing tag used to check the _stamped question")
    args = ap.parse_args()
    fails = 0

    print("== GPU ==")
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
               f"{torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info()
            ok(f"GPU memory {free / 2**30:.1f} GB free of {total / 2**30:.1f}")
            # counts_torch(grad=True) is unchunked, ~11-15 GB per call
            # regardless of batch size; the GN score needs one per iteration.
            if free / 2**30 < 16:
                warn("under 16 GB free; the GN backward is ~11-15 GB "
                     "unchunked, so this may OOM")
        else:
            bad("no CUDA device: the bias and GN stages need one")
            fails += 1
    except Exception as e:                                   # noqa: BLE001
        bad(f"torch import/CUDA query failed: {e}")
        fails += 1

    print("\n== threading ==")
    if os.environ.get("MKL_THREADING_LAYER") == "GNU":
        ok("MKL_THREADING_LAYER=GNU")
    else:
        bad("MKL_THREADING_LAYER is not GNU; torch import dies on the cluster "
            "(conda-MKL numpy vs libgomp)")
        fails += 1

    print("\n== emulator model store ==")
    store = args.store or None
    if store is None:
        try:
            from spexai.config import STORE
            store = STORE
        except Exception as e:                               # noqa: BLE001
            bad(f"cannot import spexai.config.STORE: {e}")
            fails += 1
    if store and os.path.isdir(store):
        n = len([f for f in os.listdir(store) if f.endswith(".pt")])
        ok(f"store {store} ({n} .pt model files)")
    elif store:
        bad(f"store {store} does not exist")
        fails += 1

    # THE path bug that cost a night's run. SpexTruthModel does not use
    # SPEXAI_PROCESSED directly: without an explicit datadir it derives the
    # cache location from manifest["runroot"], an ABSOLUTE path recorded on the
    # machine that TRAINED the model. The shipped manifest points at a laptop
    # directory, so on the cluster the truth stage looked for 40 GB of caches
    # under a path that does not exist. Resolve it exactly as the code will and
    # confirm the caches are actually there.
    print("\n== SPEX per-element caches (truth stage, CPU) ==")
    datadir = None
    if store and os.path.isdir(store):
        try:
            import json
            from spexai.eval import _default_datadir
            with open(os.path.join(store, "manifest.json")) as f:
                manifest = json.load(f)
            datadir = _default_datadir(store, manifest)
            src = ("SPEXAI_PROCESSED" if os.environ.get("SPEXAI_PROCESSED")
                   else f"manifest runroot {manifest.get('runroot', '')!r}")
            print(f"        resolved from {src}")
        except Exception as e:                               # noqa: BLE001
            bad(f"could not resolve the cache dir: {e}")
            fails += 1
    if datadir and os.path.isdir(datadir):
        els = [d for d in os.listdir(datadir) if d.startswith("element")]
        if els:
            ok(f"{datadir} ({len(els)} element caches)")
        else:
            bad(f"{datadir} exists but holds no element* caches")
            fails += 1
    elif datadir:
        bad(f"{datadir} does not exist. Set SPEXAI_PROCESSED (or pass "
            f"--datadir to bias_sweep) to this machine's processed/ dir")
        fails += 1

    print("\n== responses ==")
    try:
        from spexai.config import RESP_DIR
        if os.path.isdir(RESP_DIR):
            ok(f"{RESP_DIR}")
        else:
            bad(f"{RESP_DIR} does not exist; set SPEXAI_RESPONSES")
            fails += 1
    except Exception as e:                                   # noqa: BLE001
        bad(f"cannot import spexai.config.RESP_DIR: {e}")
        fails += 1

    # The question that would silently invalidate the whole run: p6_sweep
    # defaults to truth_single_n20_s3_STAMPED.npz, and nothing in the repo
    # produces a _stamped file. A fresh n100 truth will not have one, so the
    # driver points p6_sweep at the raw npz. That is only correct if the stamp
    # is provenance METADATA and not a transformation of the counts.
    print("\n== is '_stamped' metadata or a transformation? ==")
    raw = os.path.join(args.sweep_dir, f"truth_{args.ref_tag}.npz")
    stamped = os.path.join(args.sweep_dir, f"truth_{args.ref_tag}_stamped.npz")
    if not (os.path.exists(raw) and os.path.exists(stamped)):
        warn(f"cannot compare: need both {os.path.basename(raw)} and "
             f"{os.path.basename(stamped)}. Proceeding uses the RAW npz for "
             f"the new run -- verify by hand if you have not already.")
    else:
        a = np.load(raw, allow_pickle=True)["counts"]
        b = np.load(stamped, allow_pickle=True)["counts"]
        if a.shape != b.shape:
            bad(f"counts shapes differ, {a.shape} vs {b.shape}: the stamp is a "
                f"TRANSFORMATION and the raw npz must not be used")
            fails += 1
        elif np.array_equal(a, b):
            ok("counts identical -- the stamp is metadata only, raw npz is safe")
        else:
            rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)
            bad(f"counts DIFFER by {rel:.3e} relative: the stamp changes data. "
                f"STOP -- the new run would use an unstamped truth")
            fails += 1

    print("\n== cost estimate ==")
    n = args.n_points
    per = {"truth": 18.0, "bias": 262.0, "gn": 272.0}
    tot = sum(per.values()) * n
    for k, v in per.items():
        print(f"  {k:>6}: {v:>6.0f} s/pt x {n} = {v * n / 3600:>5.1f} h")
    print(f"  {'TOTAL':>6}: {tot / 3600:.1f} h")
    print("  NOTE: the 262 s/pt bias figure is a LAPTOP CPU measurement from "
          "2026-08-19.\n        On GPU it should be faster but has never been "
          "timed. Every stage is\n        resumable, so a short night still "
          "leaves usable points.")

    print(f"\n{'PREFLIGHT OK' if fails == 0 else f'PREFLIGHT FAILED ({fails})'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
