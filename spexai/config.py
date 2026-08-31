"""Package-wide path defaults, overridable by environment variable.

No campaign content lives here -- just where things are on disk. Values
mirror the machine spexai is running on: a laptop with the repo checked out,
or the cluster where the conda env + repo + data all moved to the data disk
(see the ``cluster-paths`` note in project memory).
"""
import os

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))

STORE = os.environ.get("SPEXAI_STORE", os.path.join(_PKG_ROOT, "models"))
DATADIR = os.environ.get(
    "SPEXAI_PROCESSED", os.path.expanduser("~/data/spexai_data/processed"))
RESP_DIR = os.environ.get(
    "SPEXAI_RESPONSES", os.path.expanduser("~/data/spexai_data/responses"))
RESULTS = os.environ.get(
    "SPEXAI_RESULTS", os.path.expanduser("~/data/spexai_data/results"))
