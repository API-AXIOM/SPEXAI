"""Is the measured offset a SCALED prediction, or a different direction?

    python scripts/inference/p6_linearisation_geometry.py <p6 jsonl>

``k`` is computed per parameter, which hides the question P6 is really asking.
The linearisation predicts a VECTOR ``b_sys``; the fit measures a vector
``delta``. Two very different failures both show up as "k != 1":

* **A scale error.** ``delta ~ c * b_sys`` with c != 1: same direction, wrong
  length. The screen has the physics right and the magnitude wrong, and a
  single correction factor per point would fix it.
* **A direction error.** ``delta`` points somewhere ``b_sys`` did not. No
  scalar correction can repair that, and per-parameter k values become a
  misleading summary of it.

Both are measured in units of sigma, so the components are commensurate, and
only components with a thick denominator are used -- a near-zero ``b_sys``
component contributes noise to the angle for the same reason it makes k blow
up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

THICK = 0.5          # |b_sys| / sigma required to enter the geometry


def load(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    path = Path(sys.argv[1]).expanduser()
    recs = sorted(load(path), key=lambda r: r["point"])
    design = [k for k in recs[0]["params"] if k != "point"]

    rows = []
    for r in recs:
        sig = np.asarray(r["sigma"])
        b = np.asarray(r["b_sys"]) / sig                   # (ndim,) predicted
        d = np.asarray(r["mean_delta"]) / sig              # (ndim,) measured
        m = np.abs(b) >= THICK
        if m.sum() < 2:
            continue
        bb, dd = b[m], d[m]
        nb, nd = np.linalg.norm(bb), np.linalg.norm(dd)
        cos = float(bb @ dd / (nb * nd))
        rows.append(dict(point=int(r["point"]), n=int(m.sum()),
                         ratio=float(nd / nb), cos=cos,
                         miss=float(np.linalg.norm(dd - bb)),
                         conv=bool(r["converged"]),
                         spread=float(r["start_spread_sigma"]),
                         params=r["params"]))

    print(f"{path.name}: {len(rows)} points with >=2 thick components "
          f"(|b_sys| >= {THICK} sigma)\n")
    print("ratio = |delta|/|b_sys| (1 = right length); cos = alignment "
          "(1 = right direction)")
    print(f"{'point':>5} {'n':>3} {'ratio':>7} {'cos':>7} {'miss/sig':>9} "
          f"{'spread':>8} {'conv':>5}  design")
    for q in sorted(rows, key=lambda q: -abs(q["ratio"] - 1)):
        dz = " ".join(f"{k}={q['params'][k]:g}" for k in design[:1])
        dz += " " + " ".join(f"{k}={q['params'][k]:.3g}"
                             for k in ("sigma_v", "n_h") if k in q["params"])
        print(f"{q['point']:5d} {q['n']:3d} {q['ratio']:7.3f} {q['cos']:7.4f} "
              f"{q['miss']:9.2f} {q['spread']:8.3f} {str(q['conv']):>5}  {dz}")

    good = [q for q in rows if q["spread"] < 1.0]
    rat = np.array([q["ratio"] for q in good])
    cos = np.array([q["cos"] for q in good])
    print(f"\nconverged points only ({len(good)}):")
    print(f"  ratio |delta|/|b_sys|: median {np.median(rat):.3f}, "
          f"min {rat.min():.3f}, max {rat.max():.3f}")
    print(f"  alignment cos:         median {np.median(cos):.4f}, "
          f"min {cos.min():.4f}, max {cos.max():.4f}")
    print(f"  points with cos < 0.99: "
          f"{[q['point'] for q in good if q['cos'] < 0.99]}")


if __name__ == "__main__":
    main()
