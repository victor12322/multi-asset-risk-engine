"""Portfolio risk models: parametric, historical and Monte Carlo.

All models are portfolio-level: they are fitted on an asset return matrix and a
weight vector and report VaR / CVaR as **positive loss fractions** of portfolio
value (e.g. ``0.032`` means a 3.2% loss). Multiply by the portfolio notional to
obtain currency amounts.

Models implemented
------------------
* :class:`ParametricNormalRiskModel` - variance-covariance (Gaussian) VaR with
  sample or RiskMetrics EWMA (:math:`\\lambda = 0.94`) covariance.
* :class:`HistoricalSimulationRiskModel` - empirical quantile VaR and tail
  average CVaR, with overlapping multi-day windows.
* :class:`MonteCarloNormalRiskModel` - Cholesky-simulated multivariate normal
  paths.
* :class:`MonteCarloStudentTRiskModel` - multivariate Student-t paths with
  :math:`\\nu` degrees of freedom (default 4) and optional concentrated MLE of
  :math:`\\nu`.
* :class:`FilteredEWMAStudentTRiskModel` - conditionally heteroskedastic VaR
  combining a RiskMetrics EWMA volatility filter with Student-t quantiles.

Horizon handling: parametric models scale the mean by ``T`` and volatility by
``sqrt(T)`` (square-root-of-time rule). Historical simulation uses overlapping
``T``-day returns when enough history exists; Monte Carlo explicitly simulates
``T`` daily steps per path.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Mapping, Sequence, Self

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

LOGGER = logging.getLogger(__name__)

CovarianceMethod = Literal["sample", "ewma"]
TRADING_DAYS = 252

MODEL_KEYS: tuple[str, ...] = ("normal", "historical", "mc_normal", "mc_student_t")
MODEL_LABELS: dict[str, str] = {
    "normal": "Parametric Normal (VCV)",
    "historical": "Historical Simulation",
    "mc_normal": "Monte Carlo (Normal)",
    "mc_student_t": "Monte Carlo (Student-t)",
}


class RiskModelError(ValueError):
    """Raised for invalid model parameters or inputs."""


class InsufficientDataError(RiskModelError):
    """Raised when the estimation sample is too short for the model."""


# ----------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------
def normal_quantile(confidence: float) -> float:
    """Return the loss-side standard normal quantile ``z_{1-alpha}``.

    This is the threshold return below which the worst ``1 - alpha`` fraction
    of outcomes lies, and is negative for any confidence above 50%.
    """
    return float(stats.norm.ppf(1.0 - confidence))


def normal_es_factor(confidence: float) -> float:
    """Return the Gaussian expected-shortfall multiplier ``phi(z)/(1 - alpha)``."""
    z = normal_quantile(confidence)
    return float(stats.norm.pdf(z) / (1.0 - confidence))


def ensure_positive_semidefinite(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by eigenvalue clipping.

    Args:
        matrix: Square, symmetric matrix (covariance or correlation).
        epsilon: Minimum eigenvalue retained.

    Returns:
        Symmetric positive semi-definite matrix.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise RiskModelError("Matrix must be square.")
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() >= epsilon:
        return symmetric
    clipped = np.clip(eigenvalues, epsilon, None)
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


def cholesky_factor(covariance: np.ndarray) -> np.ndarray:
    """Return a lower-triangular Cholesky factor, repairing non-PSD matrices.

    Args:
        covariance: Asset covariance matrix.

    Returns:
        Lower-triangular matrix ``L`` with ``L @ L.T`` approximately equal to
        the repaired covariance.
    """
    repaired = ensure_positive_semidefinite(covariance)
    try:
        return np.linalg.cholesky(repaired)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - repair should prevent this
        raise RiskModelError(f"Covariance matrix is not positive definite: {exc}") from exc


def sample_covariance(returns: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Return the sample covariance matrix of a ``(T, N)`` return array."""
    array = np.asarray(returns, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise RiskModelError("Returns must be a (T, N) array with T >= 2.")
    return np.cov(array, rowvar=False, ddof=ddof).reshape(array.shape[1], array.shape[1])


def ewma_covariance(returns: np.ndarray, decay: float = 0.94, bias_correct: bool = False) -> np.ndarray:
    """RiskMetrics exponentially weighted covariance of ``(T, N)`` returns.

    Uses zero-mean updates ``Sigma_t = decay * Sigma_{t-1} + (1 - decay) r_t r_t'``
    seeded with the first outer product. With ``bias_correct=True`` the result is
    rescaled by ``1 / (1 - decay**T)`` so the weights sum to one.

    Args:
        returns: Daily asset returns.
        decay: RiskMetrics decay factor (0.94 is the daily standard).
        bias_correct: Rescale to a proper weighted estimator.

    Returns:
        Symmetric PSD covariance matrix.
    """
    array = np.asarray(returns, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise RiskModelError("Returns must be a (T, N) array with T >= 2.")
    if not 0.0 < decay < 1.0:
        raise RiskModelError("decay must lie strictly between 0 and 1.")
    covariance = np.outer(array[0], array[0])
    for row in array[1:]:
        covariance = decay * covariance + (1.0 - decay) * np.outer(row, row)
    if bias_correct:
        covariance = covariance / (1.0 - decay ** array.shape[0])
    return ensure_positive_semidefinite(0.5 * (covariance + covariance.T))


def ewma_volatility(
    returns: np.ndarray | pd.Series,
    decay: float = 0.94,
    initialization: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """RiskMetrics conditional-volatility filter for a univariate return series.

    The variance recursion is

    .. math::
        \\sigma^2_{t} = \\lambda \\sigma^2_{t-1} + (1-\\lambda) r_{t-1}^2,

    seeded with the mean squared return of the first ``initialization``
    observations. The one-step-ahead forecast applies the final observation.

    Args:
        returns: Univariate return series.
        decay: RiskMetrics decay ``lambda`` (0.94 daily standard).
        initialization: Number of observations used to seed the variance.

    Returns:
        Tuple ``(sigma_series, standardized_residuals, sigma_next)``. The first
        ``initialization`` entries of ``sigma_series`` are ``NaN`` because no
        conditional estimate exists for them.

    Raises:
        RiskModelError: For invalid inputs or a degenerate (zero-variance) series.
    """
    series = np.asarray(returns, dtype=float).ravel()
    if series.size < 5:
        raise RiskModelError("The EWMA filter needs at least five observations.")
    if not 0.0 < decay < 1.0:
        raise RiskModelError("decay must lie strictly between 0 and 1.")
    if initialization < 2:
        raise RiskModelError("initialization must be at least 2.")
    if not np.isfinite(series).all():
        raise RiskModelError("Returns contain NaN or infinite values.")
    init = min(int(initialization), series.size)
    variance = float(np.mean(series[:init] ** 2))
    if variance <= 0.0:
        raise RiskModelError("Cannot filter a zero-variance return series.")
    sigma = np.full(series.size, np.nan, dtype=float)
    for t in range(init, series.size):
        variance = decay * variance + (1.0 - decay) * series[t - 1] ** 2
        sigma[t] = np.sqrt(variance)
    sigma_next = float(np.sqrt(decay * variance + (1.0 - decay) * series[-1] ** 2))
    residuals = series[init:] / sigma[init:]
    return sigma, residuals, sigma_next


def student_t_quantile(confidence: float, degrees_of_freedom: float) -> float:
    """Loss-side quantile of a unit-variance Student-t.

    SciPy's ``t`` has variance ``nu / (nu - 2)``, so the quantile is rescaled
    by ``sqrt((nu - 2) / nu)`` to make it comparable with the conditional
    volatility of the EWMA filter and with the simulated multivariate t.

    Args:
        confidence: Confidence level.
        degrees_of_freedom: Tail parameter ``nu`` (> 2).

    Returns:
        Negative return threshold for the loss tail.

    Raises:
        RiskModelError: For invalid parameters.
    """
    if not 0.5 < confidence < 1.0:
        raise RiskModelError("confidence must lie strictly between 0.5 and 1.0.")
    if degrees_of_freedom <= 2.0:
        raise RiskModelError("Student-t degrees of freedom must exceed 2.")
    raw_quantile = float(stats.t.ppf(1.0 - confidence, degrees_of_freedom))
    return float(raw_quantile * np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom))


def student_t_expected_shortfall(confidence: float, degrees_of_freedom: float) -> float:
    """Expected Shortfall multiplier of a unit-variance Student-t.

    For the loss tail beyond the quantile ``q`` the closed form is

    ``scale * f_nu(q) * (nu + q**2) / ((nu - 1) * (1 - alpha))``

    with ``scale = sqrt((nu - 2) / nu)`` and ``f_nu`` the standard t density.

    Args:
        confidence: Confidence level.
        degrees_of_freedom: Tail parameter ``nu`` (> 2).

    Returns:
        Positive Expected Shortfall per unit of volatility.

    Raises:
        RiskModelError: For invalid parameters.
    """
    if not 0.5 < confidence < 1.0:
        raise RiskModelError("confidence must lie strictly between 0.5 and 1.0.")
    if degrees_of_freedom <= 2.0:
        raise RiskModelError("Student-t degrees of freedom must exceed 2.")
    quantile = float(stats.t.ppf(1.0 - confidence, degrees_of_freedom))
    density = float(stats.t.pdf(quantile, degrees_of_freedom))
    scale = float(np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom))
    return float(
        scale
        * density
        * (degrees_of_freedom + quantile**2)
        / ((degrees_of_freedom - 1.0) * (1.0 - confidence))
    )


def portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Return ``w' Sigma w`` clipped at zero."""
    value = float(np.asarray(weights, dtype=float) @ np.asarray(covariance, dtype=float) @ np.asarray(weights, dtype=float))
    return max(value, 0.0)


def estimate_degrees_of_freedom(
    returns: np.ndarray | pd.DataFrame,
    *,
    mean: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    lower: float = 2.05,
    upper: float = 200.0,
) -> float:
    """Concentrated maximum-likelihood estimate of Student-t degrees of freedom.

    The location and scale are fixed at the sample mean and covariance, and
    :math:`\\nu` maximises

    .. math::
        \\ell(\\nu) = \\sum_t \\Big[ \\log\\Gamma\\!\\Big(\\frac{\\nu+N}{2}\\Big)
        - \\log\\Gamma\\!\\Big(\\frac{\\nu}{2}\\Big)
        - \\frac{N}{2}\\log(\\nu\\pi)
        - \\frac{\\nu+N}{2}\\log\\!\\Big(1 + \\frac{\\delta_t}{\\nu}\\Big) \\Big]

    where :math:`\\delta_t` is the squared Mahalanobis distance of observation
    ``t``. Terms independent of :math:`\\nu` are omitted.

    Args:
        returns: ``(T, N)`` asset returns.
        mean: Optional location vector (defaults to the sample mean).
        covariance: Optional scatter matrix (defaults to the ML covariance).
        lower: Lower bound for :math:`\\nu` (must exceed 2 for finite variance).
        upper: Upper bound for :math:`\\nu`.

    Returns:
        Estimated degrees of freedom.

    Raises:
        InsufficientDataError: If the sample is too small or degenerate.
    """
    array = _coerce_returns(returns)[0]
    if array.shape[0] < array.shape[1] + 5:
        raise InsufficientDataError("Too few observations to estimate Student-t degrees of freedom.")
    mu = np.asarray(mean, dtype=float) if mean is not None else array.mean(axis=0)
    scatter = np.asarray(covariance, dtype=float) if covariance is not None else sample_covariance(array, ddof=0)
    scatter = ensure_positive_semidefinite(scatter)
    try:
        inverse = np.linalg.inv(scatter)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - PSD repair
        raise InsufficientDataError("Scatter matrix is singular.") from exc
    centred = array - mu
    mahalanobis = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    mahalanobis = np.clip(mahalanobis, 1e-12, None)
    n_assets = array.shape[1]

    def negative_log_likelihood(nu: float) -> float:
        return -float(
            np.sum(
                gammaln((nu + n_assets) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * n_assets * np.log(nu * np.pi)
                - 0.5 * (nu + n_assets) * np.log1p(mahalanobis / nu)
            )
        )

    result = optimize.minimize_scalar(
        negative_log_likelihood,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:  # pragma: no cover - bounded scalar search is robust
        LOGGER.warning("Degrees-of-freedom optimisation did not converge: %s", result.message)
    return float(np.clip(result.x, lower, upper))


# ----------------------------------------------------------------------
# Input coercion
# ----------------------------------------------------------------------
def _coerce_returns(returns: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Validate and convert a return matrix to a float ndarray.

    Args:
        returns: ``(T, N)`` return matrix.

    Returns:
        Tuple ``(array, asset_names)``.

    Raises:
        RiskModelError: For non-finite, non-numeric or 1-D input.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise RiskModelError("Return matrix is empty.")
        frame = returns.astype(float)
        array = frame.to_numpy(dtype=float)
        assets = tuple(str(c) for c in frame.columns)
    else:
        array = np.asarray(returns, dtype=float)
        assets = tuple(f"asset_{i}" for i in range(array.shape[1] if array.ndim > 1 else 1))
    if array.ndim != 2:
        raise RiskModelError("Returns must be a two-dimensional (T, N) array.")
    if not np.isfinite(array).all():
        raise RiskModelError("Returns contain NaN or infinite values.")
    if array.shape[1] == 0:
        raise RiskModelError("Returns must contain at least one asset.")
    return array, assets


def _as_weights(weights: Sequence[float] | np.ndarray | pd.Series, n_assets: int) -> np.ndarray:
    """Validate a portfolio weight vector.

    Args:
        weights: Exposure vector (may sum to one, or to any positive gross
            exposure for a leveraged portfolio).
        n_assets: Expected number of assets.

    Returns:
        Float array of shape ``(n_assets,)``.

    Raises:
        RiskModelError: If the length mismatches or the vector is degenerate.
    """
    array = np.asarray(list(weights) if not isinstance(weights, np.ndarray) else weights, dtype=float).ravel()
    if array.size != n_assets:
        raise RiskModelError(f"Expected {n_assets} weights, received {array.size}.")
    if not np.isfinite(array).all():
        raise RiskModelError("Weights contain NaN or infinite values.")
    if np.allclose(array, 0.0):
        raise RiskModelError("Portfolio weights are all zero.")
    return array


@dataclass(frozen=True)
class RiskMeasureResult:
    """VaR and CVaR for one model at one confidence and horizon.

    Attributes:
        model: Human-readable model label.
        confidence: Confidence level ``alpha``.
        horizon_days: Risk horizon in trading days.
        var: Value at Risk as a positive loss fraction.
        cvar: Expected Shortfall as a positive loss fraction.
        details: Diagnostic parameters (volatility, paths, degrees of freedom).
    """

    model: str
    confidence: float
    horizon_days: int
    var: float
    cvar: float
    details: Mapping[str, float] = field(default_factory=dict)

    @property
    def tail_ratio(self) -> float:
        """CVaR / VaR, always >= 1 for a coherent risk measure."""
        return float(self.cvar / self.var) if self.var > 0 else float("nan")

    def as_row(self) -> dict[str, float | str]:
        """Flatten the result into a reporting-friendly dictionary."""
        row: dict[str, float | str] = {
            "model": self.model,
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "var": self.var,
            "cvar": self.cvar,
            "cvar_var_ratio": self.tail_ratio,
        }
        for key, value in self.details.items():
            row[key] = value
        return row


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class RiskModel(ABC):
    """Abstract portfolio risk model.

    Subclasses implement :meth:`_fit`, :meth:`var_for_weights` and
    :meth:`cvar_for_weights`. The base class validates inputs, stores the
    estimation sample and exposes a uniform :meth:`evaluate` API.
    """

    name: ClassVar[str] = "Base risk model"
    key: ClassVar[str] = "base"
    supports_analytic_euler: ClassVar[bool] = False
    min_observations: ClassVar[int] = 30

    def __init__(self, confidence: float = 0.99, horizon_days: int = 1) -> None:
        """Initialise the model.

        Args:
            confidence: Confidence level in ``(0.5, 1)``.
            horizon_days: Risk horizon in trading days (>= 1).

        Raises:
            RiskModelError: For invalid parameters.
        """
        if not 0.5 < confidence < 1.0:
            raise RiskModelError("confidence must lie strictly between 0.5 and 1.0.")
        if int(horizon_days) < 1:
            raise RiskModelError("horizon_days must be at least 1.")
        self._confidence = float(confidence)
        self._horizon = int(horizon_days)
        self._weights: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._assets: tuple[str, ...] = ()
        self._sample_size: int = 0

    # -- properties ----------------------------------------------------
    @property
    def confidence(self) -> float:
        """Confidence level ``alpha``."""
        return self._confidence

    @property
    def horizon_days(self) -> int:
        """Risk horizon in trading days."""
        return self._horizon

    @property
    def assets(self) -> tuple[str, ...]:
        """Asset identifiers seen during fitting."""
        return self._assets

    @property
    def n_assets(self) -> int:
        """Number of assets in the fitted universe."""
        return len(self._assets)

    @property
    def weights(self) -> np.ndarray:
        """Fitted portfolio weights.

        Raises:
            RiskModelError: If the model has not been fitted.
        """
        self._require_fitted()
        assert self._weights is not None
        return self._weights

    @property
    def means(self) -> np.ndarray:
        """Daily asset mean returns estimated on the sample."""
        self._require_fitted()
        assert self._means is not None
        return self._means

    @property
    def covariance(self) -> np.ndarray:
        """Daily asset covariance matrix.

        Raises:
            RiskModelError: If the model has no covariance estimate.
        """
        if self._covariance is None:
            raise RiskModelError(f"{self.name} does not expose a covariance matrix.")
        return self._covariance

    @property
    def sample_size(self) -> int:
        """Number of observations used for fitting."""
        return self._sample_size

    @property
    def fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._weights is not None

    @property
    def z_score(self) -> float:
        """Loss-side normal quantile ``Phi^-1(1 - alpha)`` (negative)."""
        return normal_quantile(self._confidence)

    # -- fitting -------------------------------------------------------
    def fit(
        self,
        returns: np.ndarray | pd.DataFrame,
        weights: Sequence[float] | np.ndarray | pd.Series,
    ) -> Self:
        """Estimate the model on ``returns`` for portfolio ``weights``.

        Args:
            returns: Daily asset returns, ``(T, N)``.
            weights: Portfolio weights aligned with the return columns.

        Returns:
            ``self``, allowing fluent use.

        Raises:
            InsufficientDataError: If the sample is too short.
            RiskModelError: For invalid inputs.
        """
        array, assets = _coerce_returns(returns)
        weight_vector = _as_weights(weights, array.shape[1])
        required = max(self.min_observations, array.shape[1] + 2)
        if array.shape[0] < required:
            raise InsufficientDataError(
                f"{self.name} requires at least {required} observations, received {array.shape[0]}."
            )
        self._assets = assets
        self._weights = weight_vector
        self._sample_size = int(array.shape[0])
        self._means = array.mean(axis=0)
        self._fit(array)
        return self

    @abstractmethod
    def _fit(self, returns: np.ndarray) -> None:
        """Subclass estimation hook; ``returns`` is a validated float array."""

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self.fitted:
            raise RiskModelError(f"{self.name} must be fitted before use.")

    # -- risk measures -------------------------------------------------
    @abstractmethod
    def var_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Value at Risk for an arbitrary weight vector.

        Args:
            weights: Portfolio weights of length :attr:`n_assets`.

        Returns:
            Positive loss fraction at the model confidence and horizon.
        """

    @abstractmethod
    def cvar_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Expected Shortfall for an arbitrary weight vector.

        Args:
            weights: Portfolio weights of length :attr:`n_assets`.

        Returns:
            Positive loss fraction at the model confidence and horizon.
        """

    def var(self) -> float:
        """Value at Risk of the fitted portfolio."""
        self._require_fitted()
        return self.var_for_weights(self.weights)

    def cvar(self) -> float:
        """Expected Shortfall of the fitted portfolio."""
        self._require_fitted()
        return self.cvar_for_weights(self.weights)

    def evaluate(self) -> RiskMeasureResult:
        """Return a :class:`RiskMeasureResult` for the fitted portfolio."""
        self._require_fitted()
        return RiskMeasureResult(
            model=self.name,
            confidence=self._confidence,
            horizon_days=self._horizon,
            var=self.var(),
            cvar=self.cvar(),
            details=self._details(),
        )

    def _details(self) -> dict[str, float]:
        """Model diagnostics attached to the risk-measure result."""
        return {"sample_size": float(self._sample_size)}

    def scenario_matrix(self, n_scenarios: int | None = None) -> np.ndarray:
        """Return horizon asset-return scenarios used for attribution.

        Args:
            n_scenarios: Optional scenario count override.

        Returns:
            ``(S, N)`` array of asset returns over the risk horizon.

        Raises:
            RiskModelError: If the subclass has no scenario representation.
        """
        raise RiskModelError(f"{self.name} does not expose a scenario matrix.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(confidence={self._confidence}, "
            f"horizon_days={self._horizon}, fitted={self.fitted})"
        )


# ----------------------------------------------------------------------
# Parametric normal
# ----------------------------------------------------------------------
class ParametricNormalRiskModel(RiskModel):
    """Variance-covariance VaR/CVaR under a multivariate normal assumption.

    .. math::
        \\mathrm{VaR}_\\alpha = -\\big(\\mu_p T + z_{1-\\alpha} \\sigma_p \\sqrt{T}\\big),
        \\qquad
        \\mathrm{CVaR}_\\alpha = -\\mu_p T + \\sigma_p \\sqrt{T}\\,
        \\frac{\\phi(z_{1-\\alpha})}{1-\\alpha}

    where :math:`\\mu_p = w'\\mu`, :math:`\\sigma_p = \\sqrt{w'\\Sigma w}`,
    :math:`z_{1-\\alpha} = \\Phi^{-1}(1-\\alpha) < 0` for confidence above 50%, and
    :math:`T` is the horizon in trading days.
    """

    name = "Parametric Normal (VCV)"
    key = "normal"
    supports_analytic_euler = True

    def __init__(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        covariance_method: CovarianceMethod = "sample",
        decay: float = 0.94,
    ) -> None:
        """Initialise the model.

        Args:
            confidence: Confidence level.
            horizon_days: Risk horizon in trading days.
            covariance_method: ``"sample"`` or ``"ewma"`` (RiskMetrics).
            decay: EWMA decay factor, used only with ``ewma``.

        Raises:
            RiskModelError: For an unsupported covariance method or decay.
        """
        super().__init__(confidence=confidence, horizon_days=horizon_days)
        if covariance_method not in ("sample", "ewma"):
            raise RiskModelError("covariance_method must be 'sample' or 'ewma'.")
        if not 0.0 < decay < 1.0:
            raise RiskModelError("decay must lie strictly between 0 and 1.")
        self.covariance_method: CovarianceMethod = covariance_method
        self.decay = float(decay)
        self._es_factor = normal_es_factor(self._confidence)

    def _fit(self, returns: np.ndarray) -> None:
        if self.covariance_method == "sample":
            self._covariance = sample_covariance(returns)
        else:
            self._covariance = ewma_covariance(returns, decay=self.decay)
        self._covariance = ensure_positive_semidefinite(self._covariance)
        assert self._means is not None
        self._means = np.asarray(self._means, dtype=float)

    def _portfolio_stats(self, weights: np.ndarray) -> tuple[float, float]:
        """Return ``(portfolio mean, portfolio volatility)`` in daily units."""
        assert self._means is not None
        mean = float(weights @ self._means)
        volatility = float(np.sqrt(portfolio_variance(weights, self.covariance)))
        return mean, volatility

    def var_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Analytical normal VaR for a weight vector."""
        self._require_fitted()
        vector = _as_weights(weights, self.n_assets)
        mean, volatility = self._portfolio_stats(vector)
        horizon = float(self.horizon_days)
        return float(-(mean * horizon + self.z_score * volatility * np.sqrt(horizon)))

    def cvar_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Analytical normal Expected Shortfall for a weight vector."""
        self._require_fitted()
        vector = _as_weights(weights, self.n_assets)
        mean, volatility = self._portfolio_stats(vector)
        horizon = float(self.horizon_days)
        return float(-mean * horizon + volatility * np.sqrt(horizon) * self._es_factor)

    def portfolio_volatility(self, weights: Sequence[float] | np.ndarray | None = None) -> float:
        """Annualised portfolio volatility (``sqrt(252)`` scaling)."""
        self._require_fitted()
        vector = self.weights if weights is None else _as_weights(weights, self.n_assets)
        _, daily = self._portfolio_stats(vector)
        return float(daily * np.sqrt(TRADING_DAYS))

    def scenario_matrix(self, n_scenarios: int | None = None) -> np.ndarray:
        """Simulate Gaussian horizon scenarios for illustration and charts."""
        self._require_fitted()
        count = int(n_scenarios or 20_000)
        assert self._means is not None
        rng = np.random.default_rng(20240917)
        factor = cholesky_factor(self.covariance)
        normal = rng.standard_normal((count, self.horizon_days, self.n_assets))
        daily = (normal @ factor.T) + self._means
        return daily.sum(axis=1)

    def _details(self) -> dict[str, float]:
        mean, volatility = self._portfolio_stats(self.weights)
        return {
            "sample_size": float(self._sample_size),
            "portfolio_mean_daily": mean,
            "portfolio_vol_daily": volatility,
            "portfolio_vol_annual": volatility * np.sqrt(TRADING_DAYS),
            "z_score": self.z_score,
            "es_factor": self._es_factor,
            "uses_ewma": 0.0 if self.covariance_method == "sample" else 1.0,
        }


# ----------------------------------------------------------------------
# Historical simulation
# ----------------------------------------------------------------------
class HistoricalSimulationRiskModel(RiskModel):
    """Non-parametric historical simulation using the empirical distribution.

    For a horizon ``T > 1`` the model uses overlapping ``T``-day cumulative
    returns when at least ``T + 30`` observations are available; otherwise it
    falls back to the square-root-of-time rule.
    """

    name = "Historical Simulation"
    key = "historical"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        use_overlapping: bool = True,
    ) -> None:
        """Initialise the model.

        Args:
            confidence: Confidence level.
            horizon_days: Risk horizon in trading days.
            use_overlapping: Use overlapping ``T``-day windows when possible.
        """
        super().__init__(confidence=confidence, horizon_days=horizon_days)
        self.use_overlapping = bool(use_overlapping)
        self.min_observations = max(30, horizon_days + 5)
        self._scenarios: np.ndarray | None = None

    def _fit(self, returns: np.ndarray) -> None:
        self._scenarios = self._build_scenarios(returns)

    def _build_scenarios(self, returns: np.ndarray) -> np.ndarray:
        horizon = self.horizon_days
        if horizon == 1:
            return returns.copy()
        if self.use_overlapping and returns.shape[0] >= horizon + 30:
            frame = pd.DataFrame(returns).rolling(horizon).sum().dropna()
            return frame.to_numpy(dtype=float)
        return returns * np.sqrt(float(horizon))

    def scenario_matrix(self, n_scenarios: int | None = None) -> np.ndarray:
        """Return the empirical horizon return scenarios (optionally tail-sampled)."""
        self._require_fitted()
        assert self._scenarios is not None
        if n_scenarios is None or n_scenarios >= self._scenarios.shape[0]:
            return self._scenarios
        losses = -(self._scenarios @ self.weights)
        order = np.argsort(losses)[::-1][: int(n_scenarios)]
        return self._scenarios[order]

    def _losses(self, weights: np.ndarray) -> np.ndarray:
        assert self._scenarios is not None
        return -(self._scenarios @ weights)

    def var_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Empirical quantile of portfolio losses."""
        self._require_fitted()
        losses = self._losses(_as_weights(weights, self.n_assets))
        return float(np.quantile(losses, self.confidence, method="linear"))

    def cvar_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Average loss conditional on exceeding the empirical VaR."""
        self._require_fitted()
        losses = self._losses(_as_weights(weights, self.n_assets))
        threshold = float(np.quantile(losses, self.confidence, method="linear"))
        tail = losses[losses >= threshold]
        if tail.size == 0:  # pragma: no cover - quantile guarantees a tail
            return threshold
        return float(tail.mean())

    def _details(self) -> dict[str, float]:
        assert self._scenarios is not None
        return {
            "sample_size": float(self._sample_size),
            "scenario_count": float(self._scenarios.shape[0]),
            "overlapping_windows": 1.0 if self.use_overlapping and self.horizon_days > 1 else 0.0,
        }


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------
class _MonteCarloRiskModel(RiskModel):
    """Shared Monte Carlo machinery (path storage and empirical risk measures)."""

    def __init__(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_paths: int = 50_000,
        seed: int = 7,
        covariance_method: CovarianceMethod = "sample",
        decay: float = 0.94,
    ) -> None:
        super().__init__(confidence=confidence, horizon_days=horizon_days)
        if covariance_method not in ("sample", "ewma"):
            raise RiskModelError("covariance_method must be 'sample' or 'ewma'.")
        if not 0.0 < decay < 1.0:
            raise RiskModelError("decay must lie strictly between 0 and 1.")
        if int(n_paths) < 1_000:
            raise RiskModelError("Monte Carlo requires at least 1,000 paths.")
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.covariance_method: CovarianceMethod = covariance_method
        self.decay = float(decay)
        self._scenarios: np.ndarray | None = None

    def _estimate_covariance(self, returns: np.ndarray) -> np.ndarray:
        if self.covariance_method == "sample":
            covariance = sample_covariance(returns)
        else:
            covariance = ewma_covariance(returns, decay=self.decay)
        return ensure_positive_semidefinite(covariance)

    def _simulate_paths(
        self,
        rng: np.random.Generator,
        n_paths: int,
        horizon_days: int,
        factor: np.ndarray,
    ) -> np.ndarray:
        """Simulate ``(n_paths, horizon_days, N)`` daily asset returns."""
        raise NotImplementedError

    def _simulate(self, factor: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Simulate and aggregate ``(n_paths, N)`` horizon returns."""
        rng = np.random.default_rng(self.seed)
        paths = self._simulate_paths(rng, self.n_paths, self.horizon_days, factor)
        return paths.sum(axis=1)

    def simulate_paths(self, n_paths: int | None = None, horizon_days: int | None = None) -> np.ndarray:
        """Simulate full daily asset-return paths.

        Unlike :meth:`scenario_matrix`, which stores only terminal horizon
        returns, this returns the intermediate daily steps required for fan
        charts and path-dependent diagnostics.

        Args:
            n_paths: Number of paths (defaults to the fitted path count).
            horizon_days: Number of daily steps (defaults to the fitted horizon).

        Returns:
            ``(n_paths, horizon_days, n_assets)`` array of daily simple returns.

        Raises:
            RiskModelError: For non-positive path or horizon counts.
        """
        self._require_fitted()
        paths = int(self.n_paths if n_paths is None else n_paths)
        horizon = int(self.horizon_days if horizon_days is None else horizon_days)
        if paths < 1 or horizon < 1:
            raise RiskModelError("Path count and horizon must be positive.")
        rng = np.random.default_rng(self.seed)
        factor = cholesky_factor(self.covariance)
        return self._simulate_paths(rng, paths, horizon, factor)

    def _fit(self, returns: np.ndarray) -> None:
        self._covariance = self._estimate_covariance(returns)
        factor = cholesky_factor(self._covariance)
        self._scenarios = self._simulate(factor, returns)

    def scenario_matrix(self, n_scenarios: int | None = None) -> np.ndarray:
        """Return simulated horizon asset returns."""
        self._require_fitted()
        assert self._scenarios is not None
        if n_scenarios is None or n_scenarios >= self._scenarios.shape[0]:
            return self._scenarios
        losses = -(self._scenarios @ self.weights)
        order = np.argsort(losses)[::-1][: int(n_scenarios)]
        return self._scenarios[order]

    def _losses(self, weights: np.ndarray) -> np.ndarray:
        assert self._scenarios is not None
        return -(self._scenarios @ weights)

    def var_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Empirical quantile of simulated portfolio losses."""
        self._require_fitted()
        losses = self._losses(_as_weights(weights, self.n_assets))
        return float(np.quantile(losses, self.confidence, method="linear"))

    def cvar_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Average simulated loss beyond the VaR threshold."""
        self._require_fitted()
        losses = self._losses(_as_weights(weights, self.n_assets))
        threshold = float(np.quantile(losses, self.confidence, method="linear"))
        tail = losses[losses >= threshold]
        if tail.size == 0:  # pragma: no cover
            return threshold
        return float(tail.mean())

    def _details(self) -> dict[str, float]:
        assert self._scenarios is not None
        return {
            "sample_size": float(self._sample_size),
            "n_paths": float(self.n_paths),
            "seed": float(self.seed),
            "scenario_count": float(self._scenarios.shape[0]),
        }


class MonteCarloNormalRiskModel(_MonteCarloRiskModel):
    """Monte Carlo VaR with multivariate normal shocks.

    Daily paths are drawn as :math:`r = \\mu + L z` with
    :math:`L L' = \\Sigma` (Cholesky), and the horizon return is the sum of
    ``T`` daily steps.
    """

    name = "Monte Carlo (Normal)"
    key = "mc_normal"

    def _simulate_paths(
        self,
        rng: np.random.Generator,
        n_paths: int,
        horizon_days: int,
        factor: np.ndarray,
    ) -> np.ndarray:
        assert self._means is not None
        normal = rng.standard_normal((n_paths, horizon_days, factor.shape[0]))
        return (normal @ factor.T) + self._means[None, None, :]

    def _details(self) -> dict[str, float]:
        details = super()._details()
        assert self._covariance is not None
        details["portfolio_vol_annual"] = float(
            np.sqrt(portfolio_variance(self.weights, self._covariance)) * np.sqrt(TRADING_DAYS)
        )
        return details


class MonteCarloStudentTRiskModel(_MonteCarloRiskModel):
    """Monte Carlo VaR with multivariate Student-t shocks.

    Shocks follow the standard construction

    .. math::
        X = \\mu + \\sqrt{\\frac{\\nu-2}{\\chi^2_\\nu}}\\; L Z,

    where one :math:`\\chi^2_\\nu` draw is shared across all assets of a path.
    The common mixing variable preserves tail dependence between assets, while
    the :math:`\\sqrt{\\nu-2}` factor keeps the simulated covariance equal to
    :math:`\\Sigma`. Degrees of freedom are fixed (default ``nu = 4``) or
    estimated by concentrated maximum likelihood.
    """

    name = "Monte Carlo (Student-t)"
    key = "mc_student_t"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_paths: int = 50_000,
        seed: int = 7,
        degrees_of_freedom: float = 4.0,
        estimate_dof: bool = False,
        covariance_method: CovarianceMethod = "sample",
        decay: float = 0.94,
    ) -> None:
        """Initialise the Student-t Monte Carlo model.

        Args:
            confidence: Confidence level.
            horizon_days: Risk horizon in trading days.
            n_paths: Number of simulated paths (>= 1,000).
            seed: Random seed for reproducibility.
            degrees_of_freedom: Tail parameter ``nu`` used when
                ``estimate_dof`` is false.
            estimate_dof: Estimate ``nu`` by concentrated MLE.
            covariance_method: ``"sample"`` or ``"ewma"`` scatter matrix.
            decay: EWMA decay factor.

        Raises:
            RiskModelError: If ``nu <= 2`` (undefined variance) or paths < 1,000.
        """
        if degrees_of_freedom <= 2.0:
            raise RiskModelError("Student-t degrees of freedom must exceed 2 for finite variance.")
        super().__init__(
            confidence=confidence,
            horizon_days=horizon_days,
            n_paths=n_paths,
            seed=seed,
            covariance_method=covariance_method,
            decay=decay,
        )
        self.configured_dof = float(degrees_of_freedom)
        self.estimate_dof = bool(estimate_dof)
        self.degrees_of_freedom = float(degrees_of_freedom)

    def _simulate(self, factor: np.ndarray, returns: np.ndarray) -> np.ndarray:
        if self.estimate_dof:
            assert self._means is not None
            self.degrees_of_freedom = estimate_degrees_of_freedom(
                returns,
                mean=self._means,
                covariance=sample_covariance(returns, ddof=0),
            )
        return super()._simulate(factor, returns)

    def _simulate_paths(
        self,
        rng: np.random.Generator,
        n_paths: int,
        horizon_days: int,
        factor: np.ndarray,
    ) -> np.ndarray:
        assert self._means is not None
        nu = float(self.degrees_of_freedom)
        normal = rng.standard_normal((n_paths, horizon_days, factor.shape[0]))
        chi_square = rng.chisquare(df=nu, size=(n_paths, 1, 1))
        mixing = np.sqrt((nu - 2.0) / chi_square)
        return (normal @ factor.T) * mixing + self._means[None, None, :]

    def _details(self) -> dict[str, float]:
        details = super()._details()
        details["degrees_of_freedom"] = float(self.degrees_of_freedom)
        details["dof_estimated"] = 1.0 if self.estimate_dof else 0.0
        assert self._covariance is not None
        details["portfolio_vol_annual"] = float(
            np.sqrt(portfolio_variance(self.weights, self._covariance)) * np.sqrt(TRADING_DAYS)
        )
        return details


class FilteredEWMAStudentTRiskModel(RiskModel):
    """Conditionally heteroskedastic Student-t VaR with a RiskMetrics filter.

    Portfolio variance follows ``sigma2[t+1] = lambda * sigma2[t] +
    (1 - lambda) * r[t]**2`` with ``lambda = 0.94``, and standardised residuals
    ``z[t] = r[t] / sigma[t]`` are used to fit the Student-t tail parameter.
    The one-step VaR is ``VaR[t+1] = -sigma[t+1] * q_unit(1 - alpha, nu)``,
    where the quantile comes from a **unit-variance** Student-t so that it is
    commensurate with both the EWMA volatility and the simulated multivariate t.
    The forecast reacts to volatility shocks within days rather than waiting for
    a 250-day sample covariance to absorb the shock.
    """

    name = "Filtered EWMA Student-t"
    key = "filtered_ewma_t"
    min_observations = 60

    def __init__(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        decay: float = 0.94,
        degrees_of_freedom: float = 4.0,
        estimate_dof: bool = True,
        initialization: int = 20,
    ) -> None:
        """Initialise the filtered EWMA Student-t model.

        Args:
            confidence: Confidence level.
            horizon_days: Risk horizon in trading days.
            decay: RiskMetrics decay factor.
            degrees_of_freedom: Fallback tail parameter ``nu``.
            estimate_dof: Estimate ``nu`` from standardised residuals.
            initialization: Observations used to seed the variance recursion.

        Raises:
            RiskModelError: For invalid decay, ``nu`` or initialization.
        """
        super().__init__(confidence=confidence, horizon_days=horizon_days)
        if not 0.0 < decay < 1.0:
            raise RiskModelError("decay must lie strictly between 0 and 1.")
        if degrees_of_freedom <= 2.0:
            raise RiskModelError("Student-t degrees of freedom must exceed 2.")
        if initialization < 2:
            raise RiskModelError("initialization must be at least 2.")
        self.decay = float(decay)
        self.configured_dof = float(degrees_of_freedom)
        self.estimate_dof = bool(estimate_dof)
        self.initialization = int(initialization)
        self.degrees_of_freedom = float(degrees_of_freedom)
        self._returns: np.ndarray | None = None
        self._cache: dict[tuple[float, ...], tuple[float, float]] = {}

    def _fit(self, returns: np.ndarray) -> None:
        self._returns = np.array(returns, dtype=float, copy=True)
        self._cache = {}
        self._conditional_state(self.weights)

    def _conditional_state(self, weights: np.ndarray) -> tuple[float, float]:
        """Return ``(next-day sigma, degrees of freedom)`` for a weight vector."""
        key = tuple(np.round(np.asarray(weights, dtype=float), 12))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        assert self._returns is not None
        portfolio = self._returns @ np.asarray(weights, dtype=float)
        _, residuals, sigma_next = ewma_volatility(
            portfolio,
            decay=self.decay,
            initialization=self.initialization,
        )
        dof = self.configured_dof
        if self.estimate_dof and residuals.size >= 30:
            try:
                dof = estimate_degrees_of_freedom(
                    residuals.reshape(-1, 1),
                    mean=np.zeros(1),
                    covariance=np.ones((1, 1)),
                    lower=2.1,
                    upper=100.0,
                )
            except (InsufficientDataError, RiskModelError):
                LOGGER.warning("EWMA filter could not estimate nu; using the configured value.")
                dof = self.configured_dof
        if key == tuple(np.round(self.weights, 12)):
            self.degrees_of_freedom = float(dof)
        state = (float(sigma_next), float(dof))
        self._cache[key] = state
        return state

    def var_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Filtered Student-t VaR for an arbitrary weight vector."""
        self._require_fitted()
        vector = _as_weights(weights, self.n_assets)
        sigma_next, dof = self._conditional_state(vector)
        quantile = student_t_quantile(self.confidence, dof)
        return float(-sigma_next * np.sqrt(self.horizon_days) * quantile)

    def cvar_for_weights(self, weights: Sequence[float] | np.ndarray) -> float:
        """Filtered Student-t Expected Shortfall for an arbitrary weight vector."""
        self._require_fitted()
        vector = _as_weights(weights, self.n_assets)
        sigma_next, dof = self._conditional_state(vector)
        return float(
            sigma_next
            * np.sqrt(self.horizon_days)
            * student_t_expected_shortfall(self.confidence, dof)
        )

    def _details(self) -> dict[str, float]:
        sigma_next, dof = self._conditional_state(self.weights)
        return {
            "sample_size": float(self._sample_size),
            "degrees_of_freedom": float(dof),
            "decay": self.decay,
            "conditional_vol_daily": sigma_next,
            "conditional_vol_annual": sigma_next * np.sqrt(TRADING_DAYS),
            "dof_estimated": 1.0 if self.estimate_dof else 0.0,
        }


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def build_risk_models(
    returns: np.ndarray | pd.DataFrame,
    weights: Sequence[float] | np.ndarray | pd.Series,
    *,
    confidence: float = 0.99,
    horizon_days: int = 1,
    covariance_method: CovarianceMethod = "sample",
    mc_paths: int = 50_000,
    seed: int = 7,
    student_t_dof: float = 4.0,
    estimate_dof: bool = False,
) -> dict[str, RiskModel]:
    """Fit the four core models on the same sample and weights.

    Args:
        returns: Daily asset returns.
        weights: Portfolio weights.
        confidence: Confidence level shared by all models.
        horizon_days: Risk horizon in trading days.
        covariance_method: Covariance estimator for parametric and MC models.
        mc_paths: Monte Carlo path count.
        seed: Monte Carlo seed.
        student_t_dof: Fixed degrees of freedom for the Student-t model.
        estimate_dof: Estimate ``nu`` by MLE.

    Returns:
        Mapping ``model_key -> fitted model`` ordered as :data:`MODEL_KEYS`.
    """
    models: dict[str, RiskModel] = {
        "normal": ParametricNormalRiskModel(
            confidence=confidence,
            horizon_days=horizon_days,
            covariance_method=covariance_method,
        ),
        "historical": HistoricalSimulationRiskModel(
            confidence=confidence,
            horizon_days=horizon_days,
        ),
        "mc_normal": MonteCarloNormalRiskModel(
            confidence=confidence,
            horizon_days=horizon_days,
            n_paths=mc_paths,
            seed=seed,
            covariance_method=covariance_method,
        ),
        "mc_student_t": MonteCarloStudentTRiskModel(
            confidence=confidence,
            horizon_days=horizon_days,
            n_paths=mc_paths,
            seed=seed,
            degrees_of_freedom=student_t_dof,
            estimate_dof=estimate_dof,
            covariance_method=covariance_method,
        ),
    }
    for model in models.values():
        model.fit(returns, weights)
    return models


def evaluate_models(models: Mapping[str, RiskModel]) -> dict[str, RiskMeasureResult]:
    """Evaluate every fitted model into a :class:`RiskMeasureResult` mapping."""
    return {key: model.evaluate() for key, model in models.items()}


def risk_measure_table(results: Mapping[str, RiskMeasureResult], portfolio_value: float = 1.0) -> pd.DataFrame:
    """Build a comparison table of VaR/CVaR in percent and currency.

    Args:
        results: Model key -> result mapping.
        portfolio_value: Notional used to convert fractions into currency.

    Returns:
        DataFrame with one row per model.
    """
    rows: list[dict[str, float | str]] = []
    for key, result in results.items():
        rows.append(
            {
                "model": result.model,
                "var_pct": 100.0 * result.var,
                "var_amount": result.var * portfolio_value,
                "cvar_pct": 100.0 * result.cvar,
                "cvar_amount": result.cvar * portfolio_value,
                "cvar_var_ratio": result.tail_ratio,
                "key": key,
            }
        )
    return pd.DataFrame(rows)
