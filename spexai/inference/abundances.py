"""Flexible abundance parametrisation for fitting.

``JointOperatorModel.flux`` (and ``SpexTruthModel``) take abundances as a plain
``{Z: solar-relative value}`` dict with H/He held solar. Real fits rarely free
every element: the classic scheme (and the thesis's) is a single global
metallicity scaling all metals, optionally with a few elements freed or tied to
a fraction of iron (an [X/Fe] ratio). ``AbundanceModel`` builds that dict from a
flat parameter dict, replacing the ad-hoc ``dict_abund`` loops in the legacy
``spexai.inference.fit``.

Build it fluently, e.g.::

    ab = (AbundanceModel(joint.elements)
          .global_metallicity("Z")          # one param scales all metals
          .free_element(26, "Fe")            # iron sampled on its own
          .tie([7, 8], "alpha_Fe", ref=26))  # N, O tied to alpha_Fe * Fe

    ab.param_names            # -> ["Z", "Fe", "alpha_Fe"]
    ab.to_abundances({"Z": 0.3, "Fe": 0.5, "alpha_Fe": 1.3})

Resolution order is: global baseline -> free (absolute) -> ties (relative to the
referenced element's already-resolved value, or absolute when ``ref`` is None).
"""
from typing import Dict, Iterable, List, Optional

import torch

# elements always held at solar and never managed here (primordial)
FIXED_SOLAR = {1, 2}

# Z -> element symbol, H through Zn (the package's 30-element store). No
# equivalent existed before this: callers spelled out bare string literals
# (e.g. priors.py's "Fe") instead of deriving them from Z.
SYMBOL: Dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn",
}


def _value(x):
    """Pass tensors through, coerce everything else to a float.

    A vectorised sampler hands in a ``(B,)`` tensor per parameter -- one value
    per walker -- and the downstream forward broadcasts it (see
    ``operator_model.abundance_weight``). ``float()`` would collapse that to a
    single walker's abundance, silently, so tensors must survive the whole
    resolution chain: global baseline, free overrides, ratio ties."""
    return x if torch.is_tensor(x) else float(x)


class AbundanceModel:
    def __init__(self, elements: Iterable[int]):
        # only metals (Z >= 3) are managed; H/He stay solar
        self.metals: List[int] = sorted(int(z) for z in elements
                                        if int(z) not in FIXED_SOLAR)
        self._global: Optional[str] = None
        self._free: Dict[int, str] = {}
        self._ties: List[tuple] = []          # (set[int], param_name, ref|None)
        self._const: List[tuple] = []         # (set[int], ratio, ref) -- no param

    # --- builders (chainable) ------------------------------------------------
    def global_metallicity(self, name: str = "Z") -> "AbundanceModel":
        """One parameter scales every metal not otherwise freed/tied."""
        self._global = name
        return self

    def free_element(self, z: int, name: Optional[str] = None) -> "AbundanceModel":
        """Element ``z`` sampled on its own, in absolute solar units."""
        z = int(z)
        self._free[z] = name or f"Z{z}"
        return self

    def tie(self, zs: Iterable[int], name: str,
            ref: Optional[int] = None) -> "AbundanceModel":
        """Tie a group of elements to a shared parameter.

        With ``ref`` set, each element's abundance is ``param * abundance[ref]``
        (an [X/ref] ratio, e.g. [alpha/Fe] with ``ref=26``); with ``ref=None``
        the parameter is the group's absolute solar abundance.
        """
        self._ties.append((set(int(z) for z in zs), name,
                           None if ref is None else int(ref)))
        return self

    def tie_const(self, zs: Iterable[int], ratio: float,
                  ref: int) -> "AbundanceModel":
        """Tie a group to ``ratio * abundance[ref]`` with **no free parameter**.

        Use for elements the literature scales with (fitted) iron but does not
        measure -- e.g. every metal not explicitly freed, tied to Fe at ratio
        1.0. ``ref`` must itself be freed or globally scaled.
        """
        self._const.append((set(int(z) for z in zs), float(ratio), int(ref)))
        return self

    # --- use -----------------------------------------------------------------
    @property
    def param_names(self) -> List[str]:
        names: List[str] = []
        if self._global:
            names.append(self._global)
        names += list(self._free.values())
        names += [name for _, name, _ in self._ties]
        return names

    def to_abundances(self, params: Dict[str, float]) -> Dict[int, float]:
        """Resolve a flat parameter dict to ``{Z: solar-relative}``.

        Values may be scalars or ``(B,)`` tensors (one per walker); the two can
        be mixed, e.g. a sampled Fe against a fixed global metallicity."""
        base = _value(params[self._global]) if self._global else 1.0
        ab: Dict[int, float] = {z: base for z in self.metals}
        for z, name in self._free.items():                 # absolute overrides
            if z in ab:
                ab[z] = _value(params[name])
        for zs, name, ref in self._ties:
            val = _value(params[name])
            scale = ab[ref] if ref is not None else 1.0
            for z in zs:
                if z in ab:
                    ab[z] = val * scale
        for zs, ratio, ref in self._const:            # constant (paramless) ties
            for z in zs:
                if z in ab:
                    ab[z] = ratio * ab[ref]
        return ab
