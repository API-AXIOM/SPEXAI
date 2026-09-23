"""One way to fit a spectrum with the emulator, with any of the samplers.

Until now there were two. ``fitting.build_posterior`` assembled a posterior for
the simulation studies, under the parameter names ``temp``/``velocity``; the
campaign scripts each re-assembled the same thing by hand under ``kT``/
``sigma_v``, because the names were baked in and so the shared builder could
not be called. The duplication was not cosmetic: it is why nine samplers were
reachable from the bake-off and only two from the user-facing path, and it is
how ``tier_c_mcmc.py`` came to build a ``VectorForward`` with no ``dem=`` while
passing DEM parameter names -- a fit that could not construct, in a copy nobody
was testing.

:class:`SpectralFit` is the single assembly point. It holds the configuration,
builds the forward once, and dispatches to
:data:`~spexai.inference.samplers.SAMPLERS`::

    from spexai.inference import SpectralFit
    from spexai.inference.priors import PriorSet, Uniform, Normal

    fit = SpectralFit(
        emulator=emu, response=resp,
        counts=counts, exposure=1.2e5,
        priors=PriorSet({
            "kT":       Uniform(1.5, 7.5),
            "Fe":       Normal(0.55, 0.05, low=0.0),   # a real measurement
            "sigma_v":  Uniform(30.0, 600.0),
            "log_norm": Uniform(10.0, 12.0),
        }),
        abundances=ab, absorption=absn, redshift=0.0173,
        band=(1.9, 12.0), device="cuda",
    )
    res = fit.sample("nautilus", n_live=2000)
    print(res.summary())

**Data comes in as arrays.** ``counts`` and ``exposure`` are passed directly
rather than wrapped in an :class:`~spexai.inference.simulate.Observation`,
because ``Observation`` carries ``true_params`` and a real observation has
none. Reading a spectrum off disk is deliberately not this package's job: load
it with astropy/sherpa/whatever and hand over the array.
:meth:`SpectralFit.from_observation` exists for the simulation studies, which
do have an ``Observation`` in hand.
"""
import warnings
from typing import Optional, Sequence

import numpy as np

from spexai.inference.units import D_REF_M

#: Legacy parameter names, and what they are now called. ``kT``/``sigma_v`` are
#: canonical: they are ``VectorForward``'s own defaults, the campaign's
#: convention and the papers'. ``temp``/``velocity`` came from ``fitting.py``
#: and are still accepted so existing parameter sets keep working.
LEGACY_NAMES = {"temp": "kT", "velocity": "sigma_v"}


def _resolve(names: Sequence[str], canonical: str, legacy: str) -> str:
    """Which spelling of a parameter this fit actually uses.

    Returns the canonical name when neither appears -- that is the DEM case,
    where ``kT`` is not sampled at all, and the forward only needs a name it can
    fail to find.
    """
    if canonical in names:
        return canonical
    if legacy in names:
        warnings.warn(
            f"parameter name {legacy!r} is deprecated; use {canonical!r}. "
            f"The old spelling still works and means the same thing.",
            DeprecationWarning, stacklevel=3)
        return legacy
    return canonical


class SpectralFit:
    """A spectrum, a model and a prior -- ready to sample.

    The forward model is built lazily, on first use of :attr:`posterior`, so
    constructing a ``SpectralFit`` is cheap and a configuration error surfaces
    where it is made rather than pages later.
    """

    def __init__(self, emulator, response, counts, exposure, priors, *,
                 abundances=None, absorption=None, dem=None,
                 redshift: float = 0.0,
                 luminosity_distance: float = D_REF_M,
                 band: Optional[Sequence[float]] = None,
                 keep: Optional[np.ndarray] = None,
                 exclude: Optional[Sequence[float]] = None,
                 velocity: Optional[float] = None,
                 fixed: Optional[dict] = None,
                 n_h_scale: Optional[float] = None,
                 device: str = "cpu",
                 **forward_kwargs):
        from spexai.inference.abundances import AbundanceModel
        from spexai.inference.response import band_mask

        if band is not None and keep is not None:
            raise ValueError("pass band=(lo, hi) or keep=<mask>, not both")
        if exclude is not None and band is None:
            raise ValueError("exclude=... only means something with band=...")

        self.emulator = emulator
        self.response = response
        self.counts = np.asarray(counts, dtype=np.float64)
        self.exposure = float(exposure)
        self.priors = priors
        self.dem = dem
        self.absorption = absorption
        self.redshift = float(redshift)
        self.luminosity_distance = float(luminosity_distance)
        self.device = device
        self.fixed = dict(fixed or {})
        self.forward_kwargs = dict(forward_kwargs)

        self.abundances = (abundances if abundances is not None
                           else AbundanceModel([]))

        n_chan = int(np.asarray(response.chan_e_cent.cpu()).size)
        if keep is not None:
            self.keep = np.asarray(keep, dtype=bool)
        elif band is not None:
            self.keep = band_mask(response, band, exclude=exclude)
        else:
            self.keep = np.ones(n_chan, dtype=bool)
        if self.keep.size != n_chan:
            raise ValueError(f"channel mask covers {self.keep.size} channels "
                             f"but the response has {n_chan}")
        if not self.keep.any():
            raise ValueError("the channel selection is empty; check band=")

        # counts may arrive either full-length or already band-restricted.
        # Both are natural: a spectrum read off disk has every channel, while
        # the campaign's cached truths are stored in-band only (dump_truth
        # writes ``d_ref[keep]``). Requiring one would make the other caller
        # scatter values back into a full array purely to have them masked out
        # again -- so the length picks the meaning, and anything else is an
        # error rather than a silent misalignment.
        n_keep = int(self.keep.sum())
        if self.counts.shape[-1] == n_chan:
            self._counts_fit = self.counts[self.keep]
        elif self.counts.shape[-1] == n_keep:
            self._counts_fit = self.counts
        else:
            raise ValueError(
                f"counts has {self.counts.shape[-1]} channels, which is "
                f"neither the response's {n_chan} nor the {n_keep} selected by "
                f"the band/keep mask")

        names = list(priors.names)
        self.temp_name = _resolve(names, "kT", "temp")
        self.velocity_name = _resolve(names, "sigma_v", "velocity")
        self.norm_name = "log_norm"
        self.nh_name = "n_h"

        # velocity is sampled when it is a parameter, otherwise pinned. An
        # explicit `velocity=` always wins, which is how a caller pins it even
        # though the emulator would happily fit it.
        if velocity is not None:
            self.velocity = float(velocity)
        elif self.velocity_name in names:
            self.velocity = None
        else:
            self.velocity = float(self.fixed.get("velocity", 0.0))

        # n_h units: the two stacks this class replaces disagreed, silently.
        # `fitting.build_posterior` passed n_h_scale=1.0, so n_h was sampled in
        # absolute cm^-2 (Uniform(0, 5e21)); the campaign relied on
        # VectorForward's default of 1e21 and sampled n_h in units of 1e21
        # (Par("n_h", p["n_h"] / 1e21, ..., 0.0, 5.0)). Picking either default
        # here would make the other convention's absorption wrong by 10^21 --
        # and wrong absorption raises no error, it just yields an absurd
        # spectrum. So when n_h actually participates, the caller must say.
        self.n_h_scale = n_h_scale
        if n_h_scale is None:
            n_h_active = (self.nh_name in names
                          or float(self.fixed.get("n_h", 0.0)) != 0.0)
            if n_h_active:
                raise ValueError(
                    f"{self.nh_name!r} is part of this fit, so n_h_scale must "
                    f"be given explicitly: pass n_h_scale=1.0 if your prior is "
                    f"in absolute cm^-2 (e.g. Uniform(0, 5e21)), or "
                    f"n_h_scale=1e21 if it is in units of 1e21 cm^-2 (e.g. "
                    f"Uniform(0, 5)). There is no safe default -- the wrong "
                    f"one is off by 10^21 and fails silently.")
            self.n_h_scale = 1.0        # unused: n_h is absent or zero

        self._forward = None
        self._posterior = None

    # --- construction -------------------------------------------------------

    @classmethod
    def from_observation(cls, obs, emulator, priors, **kw):
        """Build from a simulated :class:`~spexai.inference.simulate.Observation`.

        A convenience for the simulation studies only -- it just unpacks the
        fields. ``obs.true_params`` is deliberately ignored: a fit must not see
        the truth, and the studies that want it pass it to
        :meth:`~spexai.inference.samplers.SamplerResult.summary` themselves.
        """
        return cls(emulator=emulator, response=obs.response, counts=obs.counts,
                   exposure=obs.exposure, priors=priors, **kw)

    # --- the model ----------------------------------------------------------

    @property
    def forward(self):
        """The batched forward model, built once and cached."""
        if self._forward is None:
            from spexai.inference.vector_forward import VectorForward
            fwd = VectorForward(
                self.emulator, self.response, self.keep, list(self.priors.names),
                self.abundances, absorption=self.absorption,
                redshift=self.redshift,
                luminosity_distance=self.luminosity_distance,
                velocity=self.velocity, fixed=self.fixed,
                n_h_scale=self.n_h_scale,
                device=self.device, exposure=self.exposure,
                temp_name=self.temp_name, norm_name=self.norm_name,
                velocity_name=self.velocity_name, nh_name=self.nh_name,
                dem=self.dem, **self.forward_kwargs)
            # abundances the abundance model does not manage are constants, but
            # they still have to reach the forward; without this they silently
            # default to solar and look like a mild flux offset
            fwd.fixed.setdefault("abundances",
                                 self.fixed.get("abundances", {}))
            self._forward = fwd
        return self._forward

    @property
    def posterior(self):
        """The :class:`~spexai.inference.posterior.PoissonPosterior`.

        Exposed because the campaign scripts and the gradient paths legitimately
        want the bare object; going through :meth:`sample` is the common case,
        not the only one.
        """
        if self._posterior is None:
            from spexai.inference.posterior import PoissonPosterior
            self._posterior = PoissonPosterior(
                self.forward, self._counts_fit, self.priors)
        return self._posterior

    def pyro_model(self):
        """The same likelihood as a Pyro program, for NUTS and SVI."""
        from spexai.inference.ppl import SpectrumModel
        return SpectrumModel(self.forward, self._counts_fit,
                             self.priors.to_pyro(self.device))

    # --- sampling -----------------------------------------------------------

    def sample(self, sampler: str = "nautilus", **kw):
        """Run one sampler and return its
        :class:`~spexai.inference.samplers.SamplerResult`.

        ``sampler`` is any key of :data:`~spexai.inference.samplers.SAMPLERS`.
        Keyword arguments go straight to that sampler, so its own documented
        knobs (``n_live``, ``nwalkers``, ``nsteps``, ...) all work unchanged.
        """
        from spexai.inference import samplers as _s
        fn = _s.get_sampler(sampler)
        target = (self.pyro_model() if sampler in _s.GRADIENT_SAMPLERS
                  else self.posterior)
        return fn(target, **kw)

    @property
    def names(self):
        return list(self.priors.names)

    def __repr__(self):
        n_fit = int(self.keep.sum())
        return (f"SpectralFit({len(self.names)} params {self.names}, "
                f"{n_fit}/{self.keep.size} channels, "
                f"{self._counts_fit.sum():.3g} counts, "
                f"{'DEM' if self.dem is not None else 'single-T'}, "
                f"device={self.device!r})")
