"""Deprecated CNN-based emulator stack, kept for paper performance
comparisons against the operator formulation. Not maintained.

- neuralnetwork / dataloader / train / plot: the original per-element
  CNN/FFN emulator training code (moved out of spexai.train).
- model_cnn / fit_cnn / write_tensors_cnn: the CNN-based inference stack
  as it stood before the operator port. These are self-contained snapshots
  (they import from spexai.deprecated, not spexai.train/inference).
"""

from spexai.deprecated.neuralnetwork import *  # FFN, CNN, NonLinear
from spexai.deprecated.dataloader import *
from spexai.deprecated.plot import *
