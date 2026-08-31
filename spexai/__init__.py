"""
SpexAI

An neural network emulator for SPEX's CIE model.
"""

__version__ = "0.0.1"
__author__ = 'Jip Matthijsse'
__credits__ = 'SRON / Universiteit van Amsterdam'


# Current operator-based inference stack. The legacy CNN stack
# (spexai.inference.model / fit / write_tensors) is deprecated and no longer
# imported at package level; import those modules explicitly if you still need
# them. Plotting (fit_plots) and evaluation (spexai.eval) are not imported here
# to keep `import spexai` free of matplotlib/corner.
from spexai.inference import (JointOperatorModel, Response, fitting,
                              load_operator, simulate)

