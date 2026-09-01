"""Tier A: does per-element emulator error add coherently or in quadrature?

Every downstream bias number (Tier B's Fisher sweep, Tier C's MCMC pulls) is
computed on the FULL 30-element joint spectrum, which is automatically the
*coherent* sum of each element's error -- there is no assumption to check
there. What Tier A actually asks is a scale question: is the joint error
dominated by a handful of elements pulling the same way (so it scales like N
times a typical per-element error), or does it look more like N independent
per-element errors partly cancelling (so it scales like sqrt(N))? That number
is what calibrates how surprised to be by Tier B's per-point b_sys, and it is
cheap to get directly: compute each element's own (emulator - truth) residual
on the shared channel grid, in-band, at the SAME abundance/thermal/velocity
point, then compare the real coherent sum (= the actual joint residual) to
the quadrature combination of the same per-element residuals.

Elements outermost, one instantiation of SpexTruthModel + JointOperatorModel
per element (both support `elements=[z]`), mirroring bias_sweep.py::stage_truth
-- this is pure emulator-vs-truth in spectrum space, no fitting, no Fisher.

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python -u \\
        scripts/inference/tier_a_composition.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from campaign import (                                            # noqa: E402
    PERSEUS, find_xrism_response, band_mask, EXCLUDE_NONE, N_REF)
from bias_sweep import RANGES, sample_points, abundance_map        # noqa: E402
from spexai.config import STORE, RESULTS                          # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.response import Response                    # noqa: E402
from spexai.inference.spex_truth import SpexTruthModel            # noqa: E402


def all_points(store, response, absorption, keep, points, elements):
    """Per-point coherent-sum and quadrature-sum-of-squares residual, elements
    outermost so each element's truth+emulator models load once, not once per
    point -- the same inversion ``stage_truth`` uses and for the same reason
    (model load dominates a single-element forward's cost).

    Returns (d_total, r_coh, sumsq) each ``(n_points, n_keep)``: the running
    per-point joint truth, coherent residual sum, and sum of squared
    per-element residuals (quadrature needs only the running sum of squares,
    not every element's residual kept in memory at once).
    """
    logz = float(np.log10(PERSEUS["z"]))
    n_keep = int(keep.sum())
    d_total = np.zeros((len(points), n_keep))
    r_coh = np.zeros((len(points), n_keep))
    sumsq = np.zeros((len(points), n_keep))
    for z_el in elements:
        t0 = time.time()
        tm = SpexTruthModel(models_dir=store, elements=[z_el], device="cpu")
        em = JointOperatorModel(models_dir=store, elements=[z_el], device="cpu")
        for i, pt in enumerate(points):
            a = abundance_map(pt, [z_el])[z_el]
            common = dict(luminosity_distance=PERSEUS["dist_m"],
                         absorption=absorption, n_h=pt["n_h"] * 1e21)
            t_c = tm.predict_counts(torch.tensor([pt["kT"]]), {z_el: a}, logz,
                                    N_REF, pt["sigma_v"], response, 1.0,
                                    **common)
            e_c = em.predict_counts(torch.tensor([pt["kT"]]), {z_el: a}, logz,
                                    N_REF, pt["sigma_v"], response, 1.0,
                                    **common)
            t_c = t_c.squeeze(0).cpu().numpy()[keep]
            e_c = e_c.squeeze(0).cpu().numpy()[keep]
            r_el = e_c - t_c
            d_total[i] += t_c
            r_coh[i] += r_el
            sumsq[i] += r_el ** 2
        print(f"  Z={z_el:>2}: {len(points)} points in {time.time() - t0:.1f}s",
              flush=True)
    return d_total, r_coh, sumsq


def summarise_point(d, r_coh, sumsq, n_el, label):
    """Coherent vs quadrature combination of the per-element residuals.

    Uses the same counts-weighted FRACTIONAL residual convention as
    ``fisher_bias._diagnose_residual`` (``cw = sqrt(sum(d*r_frac**2)/sum(d))``,
    i.e. ``sqrt(sum(r**2/d)/sum(d))``) so the numbers here are directly
    comparable to the ~0.077-0.4% figures already on record elsewhere,
    rather than a raw sqrt(counts) quantity that scales with N_REF.
    """
    ok = d > 0
    cw_coh = np.sqrt(np.sum(r_coh[ok] ** 2 / d[ok]) / np.sum(d[ok]))
    cw_quad = np.sqrt(np.sum(sumsq[ok] / d[ok]) / np.sum(d[ok]))
    print(f"\n[{label}] N={n_el} elements, {ok.sum()} in-band channels")
    print(f"  counts-weighted coherent  residual (the REAL joint error): "
          f"{cw_coh:.4e}")
    print(f"  counts-weighted quadrature residual (hypothetical, random "
          f"sign): {cw_quad:.4e}")
    print(f"  ratio coherent/quadrature: {cw_coh / cw_quad:.2f}  "
          f"(1.0 = errors already look independent; "
          f"sqrt(N)={np.sqrt(n_el):.1f} = fully coherent same-sign combination)")
    return cw_coh, cw_quad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--n_points", type=int, default=4,
                    help="Perseus fiducial + N-1 LHS points from the same "
                         "cluster range Tier B sweeps")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response, exclude=EXCLUDE_NONE)
    elements = JointOperatorModel(models_dir=args.store, device="cpu").elements

    from spexai.inference.abundances import SYMBOL
    from campaign import FREE_Z
    fiducial = {"kT": PERSEUS["kT"], "sigma_v": PERSEUS["vel"],
               "n_h": PERSEUS["n_h"] / 1e21}
    for z in FREE_Z:
        fiducial[f"a_{SYMBOL[z]}"] = 1.0            # solar
    points = [fiducial]
    if args.n_points > 1:
        points += sample_points(args.n_points - 1, "single", args.seed)

    d_total, r_coh, sumsq = all_points(args.store, response, absorption, keep,
                                       points, elements)

    results = []
    for i, pt in enumerate(points):
        label = "Perseus fiducial" if i == 0 else f"sweep point {i}"
        cw_coh, cw_quad = summarise_point(d_total[i], r_coh[i], sumsq[i],
                                          len(elements), label)
        results.append(dict(point=pt, cw_coh=cw_coh, cw_quad=cw_quad))

    ratios = [r["cw_coh"] / r["cw_quad"] for r in results]
    print(f"\n=== summary over {len(results)} points ===")
    print(f"coherent/quadrature ratio: median {np.median(ratios):.2f}, "
          f"range [{min(ratios):.2f}, {max(ratios):.2f}]")
    print(f"(sqrt(N_elements)={np.sqrt(len(elements)):.1f} for reference)")

    os.makedirs(os.path.join(RESULTS, "tier_a"), exist_ok=True)
    outp = os.path.join(RESULTS, "tier_a", "composition_check.npz")
    np.savez(outp, points=[r["point"] for r in results],
             cw_coh=[r["cw_coh"] for r in results],
             cw_quad=[r["cw_quad"] for r in results], elements=elements,
             # response provenance -- see check_truth_response(): the ARF
             # rescales every channel, and nothing else stored here would
             # distinguish a flat-effective-area run from a folded one.
             rmf=os.path.basename(rmf), arf=os.path.basename(arf))
    print(f"saved {outp}")


if __name__ == "__main__":
    main()
