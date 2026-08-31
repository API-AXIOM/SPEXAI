"""Are all the packages the bake-off needs present? Run this FIRST.

Cheap insurance against the failure that motivated it: a multi-hour emcee run
completed and was then destroyed by a missing ``arviz`` at the final line,
before anything reached disk. That specific hole is closed (diagnostics can no
longer raise, and the chain now streams to HDF5), but the general lesson stands
-- check the environment before spending GPU hours, not after.

Imports every package the bake-off path actually touches and reports what is
missing, separating what would abort a run from what only degrades a
diagnostic:

    python scripts/check_deps.py
"""
import importlib
import sys

# (import name, pip name, what breaks without it)
REQUIRED = [
    ("numpy", "numpy", "everything"),
    ("scipy", "scipy", "response parsing, DEM shapes"),
    ("torch", "torch", "the emulator forward"),
    ("astropy", "astropy", "reading the RMF/ARF FITS files"),
    ("emcee", "emcee", "--sampler emcee (and the reference chain)"),
    ("h5py", "h5py", "emcee's HDF5 backend -- without it the chain is not "
                     "checkpointed and a late failure loses the run"),
    ("zeus", "zeus-mcmc", "--sampler zeus"),
    ("ultranest", "ultranest", "--sampler ultranest (and all log Z)"),
    ("pyro", "pyro-ppl", "--sampler nuts and --sampler svi"),
]

OPTIONAL = [
    ("arviz", "arviz", "ESS diagnostics; without it ESS reports NaN and the "
                       "ESS/eval column of the table is empty"),
    ("matplotlib", "matplotlib", "plotting scripts (not the bake-off itself)"),
    ("corner", "corner", "corner plots in the analysis scripts"),
]


def check(group, label):
    missing = []
    print(f"\n{label}")
    for mod, pip_name, why in group:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  OK       {mod:<12} {ver}")
        except ImportError:
            print(f"  MISSING  {mod:<12} -> {why}")
            missing.append(pip_name)
    return missing


def main():
    print(f"python {sys.version.split()[0]}\n{sys.executable}")
    hard = check(REQUIRED, "REQUIRED (a run will abort without these):")
    soft = check(OPTIONAL, "OPTIONAL (a run still completes):")

    try:
        import torch
        print(f"\ncuda available: {torch.cuda.is_available()}"
              + (f"  ({torch.cuda.get_device_name(0)})"
                 if torch.cuda.is_available() else "  -- --device cuda WILL FAIL"))
    except ImportError:
        pass

    if hard:
        print(f"\ninstall before running:\n  pip install {' '.join(hard)}")
    if soft:
        print(f"\noptional:\n  pip install {' '.join(soft)}")
    if not hard and not soft:
        print("\nall present.")
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
