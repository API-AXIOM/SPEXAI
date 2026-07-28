"""Physical normalisation constants for the SPEX-emulator forward model.

The per-element operators reproduce SPEX CIE spectra generated at the SPEX
*reference* normalisation: emission measure ``Y = 1`` (i.e. ``n_H n_e V =
1e64 m^-3``, the SPEX ``cie`` norm unit) and luminosity distance
``D_ref = 1e22 m`` (the SPEX default ``distance`` unit). Physical detector
counts therefore scale as

    counts = exposure * Y * (D_ref / D)^2 * (m^2 -> cm^2) * fold(flux)

so in ``predict_counts`` the ``norm`` argument *is* the emission measure ``Y``
in units of ``1e64 m^-3``, ``luminosity_distance`` ``D`` is in metres, and
SPEX's native SI photon flux (per m^2) is converted to the per-cm^2 convention
of the OGIP ARF (cm^2) by the factor ``FLUX_M2_TO_CM2 = 1e-4``.

The ``1e-4`` unit factor assumes the training spectra carry SPEX's SI flux
(photons s^-1 m^-2 keV^-1). Distance and emission measure are perfectly
degenerate in a single spectrum (only ``Y (D_ref/D)^2`` is constrained), so
``D`` is always supplied as a fixed input (from the known redshift/distance) and
``Y`` is the fitted quantity. The absolute scale is validated against SPEX in
``scripts/validate_spex_norm.py``.
"""

D_REF_M = 1.0e22          # SPEX reference luminosity distance (metres)
FLUX_M2_TO_CM2 = 1.0e-4   # SPEX flux is per m^2 (SI); OGIP ARF is per cm^2


def distance_factor(luminosity_distance_m: float) -> float:
    """``(D_ref / D)^2`` flux dilution relative to the SPEX reference distance."""
    return (D_REF_M / float(luminosity_distance_m)) ** 2
