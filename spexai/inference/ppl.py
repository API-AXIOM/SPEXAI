"""Pyro model over the emulator forward: priors declared, transforms free.

The same Poisson likelihood as :mod:`spexai.inference.posterior`, expressed as a
probabilistic program instead of a hand-written potential. What that buys:

* **Priors are declarative.** ``pyro.sample("kT", dist.Uniform(1.5, 7.5))``
  rather than a pair of bounds arrays, so a user can swap in a Gaussian ``n_h``
  from the literature or a log-uniform norm without touching sampler code.
* **Transforms and their Jacobians come from Pyro.** ``BoxPrior`` hand-rolls a
  scaled logit and its log-determinant; getting that term wrong biases every
  gradient-based posterior toward the middle of the box with nothing visible in
  the output. ``biject_to(constraints.interval(lo, hi))`` is the same code,
  tested upstream.
* **One model, several samplers.** SVI consumes it directly; ``initialize_model``
  turns it into a ``potential_fn`` for HMC/NUTS. Both then provably target the
  same density, which is the point of a bake-off.

**Written batch-first, deliberately.** Every sample site is scalar, so Pyro's
outer particle plate (``vectorize_particles=True``) gives each site a leading
``(P,)`` dimension; stacking them yields the ``(P, ndim)`` block that
:class:`~spexai.inference.vector_forward.VectorForward` evaluates in a single
pass. That is why SVI here does *not* forfeit the batched forward the way a
single-point NUTS does -- its Monte-Carlo particles are the batch.
"""
from typing import Dict, Mapping, Sequence

import torch


def uniform_priors(names: Sequence[str], lo, hi) -> Dict[str, "object"]:
    """``{name: Uniform(lo, hi)}`` -- the box prior, as distributions.

    Convenience for reproducing the existing box-prior fits; the model accepts
    any mapping of name to Pyro distribution, which is the point."""
    import pyro.distributions as dist
    lo = torch.as_tensor(lo, dtype=torch.float64).reshape(-1)
    hi = torch.as_tensor(hi, dtype=torch.float64).reshape(-1)
    if not (len(names) == lo.numel() == hi.numel()):
        raise ValueError("names, lo and hi must have matching lengths")
    return {n: dist.Uniform(lo[i], hi[i]) for i, n in enumerate(names)}


class SpectrumModel:
    """Pyro model: priors -> emulator counts -> Poisson likelihood.

    ``forward`` is a :class:`~spexai.inference.vector_forward.VectorForward`
    (anything with ``counts_torch(theta, grad=True)`` and a ``names`` list of
    the parameter order); ``priors`` maps each of ``forward.names`` to a Pyro
    distribution; ``data`` is the observed counts on the fitted channels.
    """

    def __init__(self, forward, data, priors: Mapping[str, "object"]):
        self.forward = forward
        self.names = list(forward.names)
        missing = [n for n in self.names if n not in priors]
        if missing:
            raise ValueError(f"no prior supplied for {missing}")
        self.priors = dict(priors)
        self.data = torch.as_tensor(data, dtype=torch.float32,
                                    device=forward.device)
        # counted in *walkers*, not calls: that is the bake-off's currency, and
        # it is what separates a sampler that batches its ensemble from one
        # that evaluates a point at a time
        self.n_eval = 0

    def __call__(self):
        """The probabilistic program. Sites are scalar so an outer particle
        plate batches them; the forward then sees ``(P, ndim)``."""
        import pyro
        import pyro.distributions as dist
        vals = [pyro.sample(n, self.priors[n]) for n in self.names]
        # each site is () or (P,) depending on the enclosing plate; a common
        # leading shape is what makes the stack a valid theta block
        theta = torch.stack(torch.broadcast_tensors(*vals), dim=-1)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)                     # (1, ndim)
        self.n_eval += int(theta.shape[0])
        mu = self.forward.counts_torch(theta, grad=True).clamp_min(1e-30)
        # to_event(1): channels are one multivariate observation, so log_prob
        # reduces over them and stays (P,) rather than (P, n_channels)
        return pyro.sample("obs", dist.Poisson(mu).to_event(1), obs=self.data)

    def potential_fn(self):
        """``(potential_fn, transforms)`` for Pyro's HMC/NUTS, derived from this
        model -- so a hand-rolled vectorised NUTS can target exactly the density
        Pyro's own NUTS would, and be checked against it."""
        from pyro.infer.mcmc.util import initialize_model
        init_params, potential, transforms, _ = initialize_model(
            self.__call__, model_args=(), num_chains=1)
        return init_params, potential, transforms
