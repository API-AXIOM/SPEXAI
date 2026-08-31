"""Standard diagnostics for an existing (unbroadened) emulator checkpoint.

Produces the same figures every training run now writes automatically
(<tag>_history.png and <tag>_spectra_T*.png; see
spexai/train/diagnostics.py) for a checkpoint trained earlier, e.g. the
HPO winners or the adaptive-study arms:

    python scripts/plot_model_diagnostics.py --ckpt <run>/t04_long.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.emulator.benchmark_operator import load_model
from spexai.train.diagnostics import run_diagnostics
from spexai.data import SpectrumData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--outdir", default=None,
                    help="default: the checkpoint's directory")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    model, variant = load_model(args.ckpt, data)
    model = model.to(device)
    tag = os.path.basename(args.ckpt).replace(".pt", "")
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.ckpt))

    history = []
    hpath = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                         f"{tag}_history.json")
    if os.path.exists(hpath):
        with open(hpath) as f:
            history = json.load(f)["history"]

    run_diagnostics(model, data, history, outdir, tag, device=device,
                    spectra=variant != "fixed_grid")


if __name__ == "__main__":
    main()
