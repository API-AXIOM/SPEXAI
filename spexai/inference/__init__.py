"""Operator-based inference stack — the current spexai pipeline.

Exposes the joint emulator, instrument response, simulation and fitting. The
legacy CNN stack (``model.py``, ``fit.py``, ``write_tensors.py``,
``model_old.py``, ``fit_old.py``) is deprecated and is no longer imported here;
import those modules explicitly if you still need them. Plotting
(``fit_plots``) and evaluation (``spexai.eval``) are intentionally left out of
this import so ``import spexai`` does not pull in matplotlib/corner.

**Import order matters, and this module fixes it.** PyTorch bundles its own
copy of ``libomp``; conda-forge NumPy and SciPy link the environment's. NumPy
is the one that claims it first, via its BLAS, and once it has, torch and SciPy
both find it already initialised and are content. Import torch before NumPy and
torch's bundled copy wins the race instead, so the next library to load the
environment's copy aborts the process with ``OMP: Error #15`` (Abort trap 6).
Measured in this env (torch 2.13.0, numpy 2.5.1)::

    import numpy; import torch; import scipy.sparse   # ok
    import torch;  import scipy.sparse                # Abort trap 6

Importing NumPy first here makes every ``spexai.inference`` entry point safe
without the ``KMP_DUPLICATE_LIB_OK=TRUE`` workaround. That variable does not
fix the conflict; it only silences a warning that explicitly says it "may cause
crashes or silently produce incorrect results", which is the wrong trade for a
package whose purpose is numerical calibration.

This is a load-order fix, so it holds only while nothing has already imported
torch. A user who runs ``import torch`` and then ``import spexai.inference``
will still hit the abort; the fix for that is a single OpenMP runtime in the
environment, not an environment variable.
"""
import numpy   # noqa: F401  -- MUST precede torch; see the note above

from spexai.inference.operator_model import JointOperatorModel, load_operator
from spexai.inference.response import Response
from spexai.inference.abundances import AbundanceModel
from spexai.inference.spex_truth import SpexTruthModel
from spexai.inference.absorption import Absorption
from spexai.inference.units import D_REF_M, FLUX_M2_TO_CM2
from spexai.inference.priors import PriorSet, Uniform, LogUniform, Normal
from spexai.inference.spectral_fit import SpectralFit
from spexai.inference import (simulate, fitting, tempdist, abundances,
                              spex_truth, absorption, units, priors, samplers)

__all__ = ["JointOperatorModel", "load_operator", "Response", "AbundanceModel",
           "SpexTruthModel", "Absorption", "D_REF_M", "FLUX_M2_TO_CM2",
           "SpectralFit", "PriorSet", "Uniform", "LogUniform", "Normal",
           "simulate", "fitting", "tempdist", "abundances", "spex_truth",
           "absorption", "units", "priors", "samplers"]
