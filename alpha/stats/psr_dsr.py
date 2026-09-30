"""Probabilistic Sharpe, Deflated Sharpe and Minimum Track Record Length.

Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012) and
"The Deflated Sharpe Ratio" (2014).

All Sharpe ratios here are PER PERIOD (not annualised) and use a zero risk-free
rate, matching `QUANT_PHILOSOPHY.md`. For daily data, divide an annualised
Sharpe by sqrt(252) before calling these functions. γ3 is skewness and γ4 is
Pearson (non-excess) kurtosis of the returns; a normal distribution has γ4 = 3.

    σ(SR̂)       = sqrt(1 − γ3·SR̂ + (γ4 − 1)/4 · SR̂²)
    PSR(SR*)    = Φ( (SR̂ − SR*) · sqrt(T − 1) / σ(SR̂) )

    SR*_0       = sqrt(V[SR̂_n]) · E[max of N standard normals]
    DSR         = PSR(SR*_0)

E[max] is computed exactly, E[max] = ∫ z · N·φ(z)·Φ(z)^(N−1) dz, which is valid
for the fractional N that the effective-trial estimate produces. The paper's
closed-form approximation (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)), γ ≈ 0.5772,
is kept as `bailey_approx_expected_max_z` for parity with the paper only: it is
negative for N < 1.28, which would turn a deflation into a bonus for highly
correlated trial families.

    MinTRL      = 1 + σ(SR̂)² · ( z_c / (SR̂ − SR*) )²

V[SR̂_n] is the variance of the per-period Sharpe ratios of the N trials, and N
is the effective number of independent trials (see `alpha.scout.trials`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats

EULER_MASCHERONI_FLOAT = 0.5772156649015329


@dataclass(frozen=True)
class SharpeMoments:
    sharpe_float: float
    skewness_float: float
    kurtosis_float: float
    observation_count_int: int


def sharpe_moments(return_vec) -> SharpeMoments:
    """Per-period Sharpe (mean / sample std), skewness and Pearson kurtosis. NaN is dropped; infinity raises."""
    return_arr = np.asarray(return_vec, dtype=float)
    if np.isinf(return_arr).any():
        raise ValueError("Returns contain infinity; fix the series instead of dropping it.")
    return_arr = return_arr[~np.isnan(return_arr)]
    if return_arr.size < 3:
        raise ValueError("Need at least three finite returns.")
    std_float = float(return_arr.std(ddof=1))
    if std_float <= 0.0:
        raise ValueError("Returns have zero dispersion.")
    return SharpeMoments(
        sharpe_float=float(return_arr.mean()) / std_float,
        skewness_float=float(stats.skew(return_arr)),
        kurtosis_float=float(stats.kurtosis(return_arr, fisher=False)),
        observation_count_int=int(return_arr.size),
    )


def _sharpe_dispersion_float(sharpe_float: float, skewness_float: float, kurtosis_float: float) -> float:
    variance_factor_float = 1.0 - skewness_float * sharpe_float + (kurtosis_float - 1.0) / 4.0 * sharpe_float**2
    if variance_factor_float <= 0.0:
        raise ValueError("Non-positive Sharpe variance factor; skew/kurtosis are inconsistent with this Sharpe.")
    return float(np.sqrt(variance_factor_float))


def probabilistic_sharpe_ratio(
    sharpe_float: float,
    observation_count_int: int,
    skewness_float: float,
    kurtosis_float: float,
    benchmark_sharpe_float: float = 0.0,
) -> float:
    """Probability that the true per-period Sharpe exceeds `benchmark_sharpe_float`."""
    if observation_count_int < 2:
        raise ValueError("observation_count_int must be >= 2.")
    dispersion_float = _sharpe_dispersion_float(sharpe_float, skewness_float, kurtosis_float)
    z_float = (sharpe_float - benchmark_sharpe_float) * np.sqrt(observation_count_int - 1.0) / dispersion_float
    return float(stats.norm.cdf(z_float))


def bailey_approx_expected_max_z(trial_count_float: float) -> float:
    """Closed-form approximation of E[max of N standard normals] from the DSR paper (valid for large N only)."""
    n_float = float(trial_count_float)
    return float(
        (1.0 - EULER_MASCHERONI_FLOAT) * stats.norm.ppf(1.0 - 1.0 / n_float)
        + EULER_MASCHERONI_FLOAT * stats.norm.ppf(1.0 - 1.0 / (n_float * np.e))
    )


def exact_expected_max_z(trial_count_float: float) -> float:
    """E[max of N i.i.d. standard normals] by numerical integration; N may be fractional (N >= 1)."""
    n_float = float(trial_count_float)
    if n_float < 1.0:
        raise ValueError("trial_count_float must be >= 1.")

    def integrand_float(z_float: float) -> float:
        return z_float * n_float * stats.norm.pdf(z_float) * stats.norm.cdf(z_float) ** (n_float - 1.0)

    value_float, _ = integrate.quad(integrand_float, -12.0, 12.0, limit=200)
    return float(value_float)


def expected_max_sharpe_ratio(trial_sharpe_variance_float: float, effective_trial_count_float: float) -> float:
    """Expected maximum per-period Sharpe of N unskilled trials (the DSR benchmark SR*_0).

    Uses the exact E[max]; it is 0 at N = 1 (no selection) and grows smoothly with N.
    """
    if trial_sharpe_variance_float < 0.0:
        raise ValueError("trial_sharpe_variance_float must be >= 0.")
    if effective_trial_count_float < 1.0:
        raise ValueError("effective_trial_count_float must be >= 1.")
    return float(np.sqrt(trial_sharpe_variance_float) * exact_expected_max_z(effective_trial_count_float))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    deflated_sharpe_float: float
    benchmark_sharpe_float: float
    moments: SharpeMoments
    effective_trial_count_float: float
    trial_sharpe_variance_float: float


def deflated_sharpe_ratio(
    return_vec,
    trial_sharpe_variance_float: float | None,
    effective_trial_count_float: float,
) -> DeflatedSharpeResult:
    """DSR of the selected strategy's returns, given the family's trial statistics.

    `trial_sharpe_variance_float=None` means the family's Sharpe dispersion is
    unknown (fewer than two recorded trials). The null sampling variance of a
    Sharpe estimate, 1 / (T − 1), is used instead: even unskilled trials differ
    by at least that much, so the deflation never silently drops to zero.
    """
    moments = sharpe_moments(return_vec)
    if trial_sharpe_variance_float is None:
        trial_sharpe_variance_float = 1.0 / (moments.observation_count_int - 1.0)
    benchmark_sharpe_float = expected_max_sharpe_ratio(trial_sharpe_variance_float, effective_trial_count_float)
    return DeflatedSharpeResult(
        deflated_sharpe_float=probabilistic_sharpe_ratio(
            sharpe_float=moments.sharpe_float,
            observation_count_int=moments.observation_count_int,
            skewness_float=moments.skewness_float,
            kurtosis_float=moments.kurtosis_float,
            benchmark_sharpe_float=benchmark_sharpe_float,
        ),
        benchmark_sharpe_float=benchmark_sharpe_float,
        moments=moments,
        effective_trial_count_float=float(effective_trial_count_float),
        trial_sharpe_variance_float=float(trial_sharpe_variance_float),
    )


def minimum_track_record_length(
    sharpe_float: float,
    skewness_float: float,
    kurtosis_float: float,
    benchmark_sharpe_float: float = 0.0,
    confidence_float: float = 0.95,
) -> float:
    """Observations needed for PSR(benchmark) to reach `confidence_float`.

    Returns +inf when the observed Sharpe does not exceed the benchmark: no track
    record length can confirm it.
    """
    if not 0.0 < confidence_float < 1.0:
        raise ValueError("confidence_float must be in (0, 1).")
    if sharpe_float <= benchmark_sharpe_float:
        return float("inf")
    dispersion_float = _sharpe_dispersion_float(sharpe_float, skewness_float, kurtosis_float)
    z_float = stats.norm.ppf(confidence_float)
    return float(1.0 + dispersion_float**2 * (z_float / (sharpe_float - benchmark_sharpe_float)) ** 2)
