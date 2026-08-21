"""Do the campaign's data paths actually resolve on this machine? Run this
FIRST when reproducing the hot-floor campaign (or the sampler bake-off, SBC,
Perseus showcase, ...) somewhere new -- a laptop set up to develop the
emulator, or a fresh cluster account.

The four ``SPEXAI_*`` paths (see ``spexai/config.py``) default to
``~/data/spexai_data/...``, which matches the cluster layout but not every
laptop -- see the ``cluster-paths`` note in project memory for a machine that
keeps the same data under a differently-named tree instead. A mismatch here
is a configuration gap (export the right env vars, or scope them to the
conda env with ``conda env config vars set -n <env> ...``), not a code bug.

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python scripts/inference/check_paths.py

Package-level dependencies (torch, emcee, ultranest, ...) are
``scripts/check_deps.py``'s job, not this script's -- this only checks paths.
It imports ``spexai.config`` rather than ``campaign.py`` to read the four
paths, which is simpler and skips the extra ``sys.path`` wiring, but note
``import spexai`` pulls in torch regardless (via ``spexai/__init__.py``), so
this still needs the full conda env, not a bare Python.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from spexai.config import STORE, DATADIR, RESP_DIR, RESULTS  # noqa: E402

# Same defaults as scripts/inference/campaign.py's RESOLVE_RMF/RESOLVE_ARF --
# read directly here rather than importing campaign.py, which pulls in torch.
RESOLVE_RMF = os.environ.get("SPEXAI_RESOLVE_RMF", "rsl_Hp_L_2025.rmf")
RESOLVE_ARF = os.environ.get("SPEXAI_RESOLVE_ARF", "rsl_extflat5_GVC_2025.arf")


def _check_store():
    """Model store: needed by every campaign script (emulator element set)."""
    manifest = os.path.join(STORE, "manifest.json")
    if not os.path.isdir(STORE):
        return False, f"MISSING  directory does not exist: {STORE}"
    if not os.path.exists(manifest):
        return False, f"MISSING  no manifest.json under {STORE}"
    with open(manifest) as f:
        n = len(json.load(f)["elements"])
    return True, f"OK       {n} elements  ({STORE})"


def _check_processed():
    """Per-element SPEX truth caches: needed by dump_truth.py, fisher_bias.py,
    check_dem.py and bias_sweep.py's truth stage when run locally (not needed
    on the cluster MCMC path, which consumes a precomputed truth npz)."""
    if not os.path.isdir(DATADIR):
        return False, f"MISSING  directory does not exist: {DATADIR}"
    n = sum(1 for d in os.listdir(DATADIR) if d.startswith("element"))
    if n == 0:
        return False, f"EMPTY    no element* caches under {DATADIR}"
    return True, f"OK       {n} element caches  ({DATADIR})"


def _check_responses():
    """XRISM/Resolve RMF+ARF: needed by every campaign script (the response
    the truth and the fit are both folded through)."""
    if not os.path.isdir(RESP_DIR):
        return False, f"MISSING  directory does not exist: {RESP_DIR}"
    missing = [n for n in (RESOLVE_RMF, RESOLVE_ARF)
              if not os.path.exists(os.path.join(RESP_DIR, n))]
    if missing:
        return False, f"MISSING  {missing} not found under {RESP_DIR}"
    return True, f"OK       {RESOLVE_RMF}, {RESOLVE_ARF}  ({RESP_DIR})"


def _check_results():
    """Output dir: doesn't need to pre-exist, just needs to be creatable."""
    try:
        os.makedirs(RESULTS, exist_ok=True)
        probe = os.path.join(RESULTS, ".check_paths_probe")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
    except OSError as e:
        return False, f"NOT WRITABLE  {RESULTS}: {e}"
    return True, f"OK       writable  ({RESULTS})"


def main():
    checks = [
        ("SPEXAI_STORE (model store)", _check_store),
        ("SPEXAI_PROCESSED (SPEX truth caches)", _check_processed),
        ("SPEXAI_RESPONSES (XRISM/Resolve RMF+ARF)", _check_responses),
        ("SPEXAI_RESULTS (output dir)", _check_results),
    ]
    hard_fail = False
    for label, fn in checks:
        ok, msg = fn()
        print(f"{label}:\n  {msg}\n")
        if not ok:
            hard_fail = True

    if hard_fail:
        print("Some paths are missing above. If this machine's data lives "
              "somewhere other than the defaults shown, scope the right\n"
              "SPEXAI_* env vars to this conda environment rather than "
              "exporting them per-shell, e.g.:\n"
              "  conda env config vars set -n <env> SPEXAI_PROCESSED=... "
              "SPEXAI_RESPONSES=... SPEXAI_RESULTS=...")
    else:
        print("all paths resolve.")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
