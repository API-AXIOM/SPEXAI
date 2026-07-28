"""Operator-based inference stack — the current spexai pipeline.

Exposes the joint emulator, instrument response, simulation and fitting. The
legacy CNN stack (``model.py``, ``fit.py``, ``write_tensors.py``,
``model_old.py``, ``fit_old.py``) is deprecated and is no longer imported here;
import those modules explicitly if you still need them. Plotting
(``fit_plots``) and evaluation (``spexai.eval``) are intentionally left out of
this import so ``import spexai`` does not pull in matplotlib/corner.
"""
from spexai.inference.operator_model import JointOperatorModel, load_operator
from spexai.inference.response import Response
from spexai.inference.abundances import AbundanceModel
from spexai.inference.spex_truth import SpexTruthModel
from spexai.inference.absorption import Absorption
from spexai.inference.units import D_REF_M, FLUX_M2_TO_CM2
from spexai.inference import (simulate, fitting, tempdist, abundances,
                              spex_truth, absorption, units)

__all__ = ["JointOperatorModel", "load_operator", "Response", "AbundanceModel",
           "SpexTruthModel", "Absorption", "D_REF_M", "FLUX_M2_TO_CM2",
           "simulate", "fitting", "tempdist", "abundances", "spex_truth",
           "absorption", "units"]
