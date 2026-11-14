"""Euler risk attribution: Marginal VaR, Component VaR, Incremental VaR and CVaR.

Risk measures are positively homogeneous of degree one in portfolio weights, so
Euler's theorem gives an exact additive decomposition:

.. math::
    \\mathrm{VaR}(w) = \\sum_{i=1}^{n} w_i \\frac{\\partial \\mathrm{VaR}}{\\partial w_i}
    = \\sum_{i=1}^{n} \\mathrm{CVaR}_i^{\\text{component}}

For the parametric normal model the gradient is available in closed form, so the
identity holds to machine precision. For historical and Monte Carlo models the
gradient is estimated by central differences on the model's own scenario set and
the additive reconstruction error is reported explicitly.

Conventions: risk numbers are positive loss fractions; weights are portfolio
exposures (they need not sum to one, but gross exposure must be positive).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .risk_models import (
    RiskModel,
    RiskModelError,
    _as_weights,
    portfolio_variance,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "SPY": 0.40,
    "QQQ": 0.20,
    "TLT": 0.20,
    "GLD": 0.10,
    "EURUSD=X": 0.10,
}

EULER_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RiskDecomposition:
    """Additive risk attribution for a fitted portfolio.

    Attributes:
        tickers: Asset identifiers aligned with every array.
        weights: Portfolio weights.
        marginal_var: ``d VaR / d w_i`` (loss fraction per unit weight).
        component_var: ``w_i * marginal_var_i``; sums to total VaR for a
            homogeneous risk measure.
        incremental_var: Exact VaR change from removing position ``i``
            (positive means the position adds risk).
        component_cvar: Contribution to Expected Shortfall.
        marginal_cvar: ``d CVaR / d w_i``.
        total_var: Portfolio VaR at the fitted weights.
        total_cvar: Portfolio CVaR at the fitted weights.
        method: Description of the attribution algorithm.
        model_name: Name of the underlying risk model.
    """

    tickers: tuple[str, ...]
    weights: np.ndarray
    marginal_var: np.ndarray
    component_var: np.ndarray
    incremental_var: np.ndarray
    component_cvar: np.ndarray
    marginal_cvar: np.ndarray
    total_var: float
    total_cvar: float
    method: str
    model_name: str

    @property
    def component_var_sum(self) -> float:
        """Sum of component VaR contributions."""
        return float(np.sum(self.component_var))

    @property
    def component_cvar_sum(self) -> float:
        """Sum of component CVaR contributions."""
        return float(np.sum(self.component_cvar))

    @property
    def euler_error(self) -> float:
        """Absolute additive reconstruction error for VaR."""
        return float(abs(self.component_var_sum - self.total_var))

    @property
    def cvar_euler_error(self) -> float:
        """Absolute additive reconstruction error for CVaR."""
        return float(abs(self.component_cvar_sum - self.total_cvar))

    @property
    def euler_identity_holds(self) -> bool:
        """Whether VaR reconstruction is within :data:`EULER_TOLERANCE`."""
        return self.euler_error <= EULER_TOLERANCE

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute weights."""
        return float(np.sum(np.abs(self.weights)))

    def component_share(self) -> np.ndarray:
        """Component VaR as a share of total VaR (zeros when VaR is zero)."""
        if abs(self.total_var) < 1e-15:
            return np.zeros_like(self.weights)
        return self.component_var / self.total_var

    def cvar_share(self) -> np.ndarray:
        """Component CVaR as a share of total CVaR (zeros when CVaR is zero)."""
        if abs(self.total_cvar) < 1e-15:
            return np.zeros_like(self.weights)
        return self.component_cvar / self.total_cvar

    def to_frame(self) -> pd.DataFrame:
        """Return the attribution table indexed by ticker."""
        return pd.DataFrame(
            {
                "weight": self.weights,
                "marginal_var": self.marginal_var,
                "component_var": self.component_var,
                "component_share": self.component_share(),
                "incremental_var": self.incremental_var,
                "component_cvar": self.component_cvar,
                "cvar_share": self.cvar_share(),
            },
            index=pd.Index(self.tickers, name="asset"),
        )

    def totals_row(self) -> pd.Series:
        """Return a totals row consistent with :meth:`to_frame`."""
        return pd.Series(
            {
                "weight": float(np.sum(self.weights)),
                "marginal_var": float(np.nan),
                "component_var": self.component_var_sum,
                "component_share": float(np.sum(self.component_share())),
                "incremental_var": float(np.sum(self.incremental_var)),
                "component_cvar": self.component_cvar_sum,
                "cvar_share": float(np.sum(self.cvar_share())),
            },
            name="TOTAL",
        )


# ----------------------------------------------------------------------
# Closed-form Euler decomposition for the normal model
# ----------------------------------------------------------------------
def analytic_normal_euler(
    means: np.ndarray,
    covariance: np.ndarray,
    weights: Sequence[float] | np.ndarray,
    confidence: float,
    horizon_days: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form Euler decomposition of normal VaR and CVaR.

    With :math:`\\mathrm{VaR} = -(\\mu_p T + z_{1-\\alpha}\\sigma_p\\sqrt{T})`
    and :math:`z_{1-\\alpha} = \\Phi^{-1}(1-\\alpha) < 0`,

    .. math::
        \\frac{\\partial \\mathrm{VaR}}{\\partial w_i}
        = -T\\mu_i - z_{1-\\alpha}\\sqrt{T}\\,\\frac{(\\Sigma w)_i}{\\sigma_p},

    and with :math:`k = \\phi(z_{1-\\alpha})/(1-\\alpha)`,

    .. math::
        \\frac{\\partial \\mathrm{CVaR}}{\\partial w_i}
        = -T\\mu_i + k\\sqrt{T}\\,\\frac{(\\Sigma w)_i}{\\sigma_p}.

    Args:
        means: Daily asset mean returns.
        covariance: Daily asset covariance matrix.
        weights: Portfolio weights.
        confidence: Confidence level.
        horizon_days: Risk horizon in trading days.

    Returns:
        Tuple ``(marginal_var, component_var, marginal_cvar, component_cvar)``.

    Raises:
        RiskModelError: If dimensions are inconsistent.
    """
    mu = np.asarray(means, dtype=float)
    sigma_matrix = np.asarray(covariance, dtype=float)
    weight_vector = np.asarray(weights, dtype=float).ravel()
    if mu.size != weight_vector.size or sigma_matrix.shape != (mu.size, mu.size):
        raise RiskModelError("Inconsistent dimensions in analytic Euler decomposition.")
    horizon = float(horizon_days)
    z = float(stats.norm.ppf(1.0 - confidence))
    es_factor = float(stats.norm.pdf(z) / (1.0 - confidence))
    sigma_p = float(np.sqrt(portfolio_variance(weight_vector, sigma_matrix)))
    if sigma_p <= 1e-15:
        zeros = np.zeros_like(weight_vector)
        return zeros, zeros, zeros, zeros

    gradient = sigma_matrix @ weight_vector / sigma_p
    marginal_var = -horizon * mu - z * np.sqrt(horizon) * gradient
    marginal_cvar = -horizon * mu + es_factor * np.sqrt(horizon) * gradient
    component_var = weight_vector * marginal_var
    component_cvar = weight_vector * marginal_cvar
    return marginal_var, component_var, marginal_cvar, component_cvar


# ----------------------------------------------------------------------
# Scenario-based attribution
# ----------------------------------------------------------------------
def empirical_component_cvar(
    scenarios: np.ndarray,
    weights: Sequence[float] | np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Tail-expectation Euler decomposition of scenario-based CVaR.

    For the tail set :math:`\\mathcal{T} = \\{s : L_s \\ge \\mathrm{VaR}\\}`
    where :math:`L_s = -w'r_s`,

    .. math::
        \\mathrm{CVaR}_i^{\\text{comp}} = -\\frac{1}{|\\mathcal{T}|}
        \\sum_{s \\in \\mathcal{T}} w_i r_{s,i},

    which sums exactly to the empirical Expected Shortfall.

    Args:
        scenarios: ``(S, N)`` horizon asset returns.
        weights: Portfolio weights.
        confidence: Confidence level defining the tail.

    Returns:
        Tuple ``(component_cvar, tail_losses, total_cvar)`` where
        ``tail_losses`` are the losses in the tail set.
    """
    matrix = np.asarray(scenarios, dtype=float)
    weight_vector = np.asarray(weights, dtype=float).ravel()
    if matrix.ndim != 2 or matrix.shape[1] != weight_vector.size:
        raise RiskModelError("Scenario matrix and weights have inconsistent dimensions.")
    losses = -(matrix @ weight_vector)
    threshold = float(np.quantile(losses, confidence, method="linear"))
    mask = losses >= threshold
    if not mask.any():  # pragma: no cover - quantile always yields a tail
        return np.zeros_like(weight_vector), losses, threshold
    tail = matrix[mask]
    contributions = -(tail * weight_vector).mean(axis=0)
    return contributions, losses[mask], float(losses[mask].mean())


def numerical_marginal_var(
    model: RiskModel,
    weights: Sequence[float] | np.ndarray,
    step: float = 1e-4,
) -> np.ndarray:
    """Central-difference gradient of the model's VaR with respect to weights.

    Args:
        model: A fitted risk model.
        weights: Evaluation point.
        step: Absolute weight perturbation.

    Returns:
        Array of marginal VaR values.
    """
    base = _as_weights(weights, model.n_assets)
    gradient = np.zeros(model.n_assets, dtype=float)
    for index in range(model.n_assets):
        up = base.copy()
        down = base.copy()
        up[index] += step
        down[index] -= step
        gradient[index] = (model.var_for_weights(up) - model.var_for_weights(down)) / (2.0 * step)
    return gradient


class EulerRiskDecomposer:
    """Compute Euler risk attribution for any fitted :class:`RiskModel`.

    The decomposer uses the analytic normal gradient when available and falls
    back to scenario-based tail expectations for simulation models. Incremental
    VaR is always an exact repricing difference on the model's distribution.

    Example:
        >>> from risk_engine.risk_models import ParametricNormalRiskModel
        >>> model = ParametricNormalRiskModel(0.99).fit(returns, weights)
        >>> decomposition = EulerRiskDecomposer(model).decompose()
        >>> decomposition.euler_identity_holds
        True
    """

    def __init__(self, model: RiskModel) -> None:
        """Store the fitted model.

        Args:
            model: A fitted risk model.

        Raises:
            RiskModelError: If the model has not been fitted.
        """
        if not model.fitted:
            raise RiskModelError("EulerRiskDecomposer requires a fitted model.")
        self.model = model

    def incremental_var_for_adjustment(self, asset_index: int, delta_weight: float) -> float:
        """Exact VaR change from adjusting one position by ``delta_weight``.

        Args:
            asset_index: Position index.
            delta_weight: Additive weight change (negative removes exposure).

        Returns:
            ``VaR(w + delta e_i) - VaR(w)`` under the same fitted distribution.

        Raises:
            RiskModelError: If the index is out of range.
        """
        if not 0 <= asset_index < self.model.n_assets:
            raise RiskModelError("asset_index is out of range.")
        base = self.model.weights
        adjusted = base.copy()
        adjusted[asset_index] += float(delta_weight)
        return float(self.model.var_for_weights(adjusted) - self.model.var_for_weights(base))

    def decompose(self) -> RiskDecomposition:
        """Compute the full attribution table.

        Returns:
            A :class:`RiskDecomposition` with marginal, component, incremental
            and CVaR contributions.
        """
        model = self.model
        weights = model.weights
        total_var = float(model.var_for_weights(weights))
        total_cvar = float(model.cvar_for_weights(weights))

        if model.supports_analytic_euler:
            marginal_var, component_var, marginal_cvar, component_cvar = analytic_normal_euler(
                model.means,
                model.covariance,
                weights,
                model.confidence,
                model.horizon_days,
            )
            method = "Analytic Euler (Gaussian gradient)"
        else:
            marginal_var = numerical_marginal_var(model, weights)
            component_var = weights * marginal_var
            try:
                scenarios = model.scenario_matrix()
            except RiskModelError as exc:  # pragma: no cover - all models expose scenarios
                raise RiskModelError(
                    f"{model.name} cannot be decomposed: no scenario matrix available."
                ) from exc
            component_cvar, _, empirical_cvar = empirical_component_cvar(
                scenarios, weights, model.confidence
            )
            total_cvar = float(empirical_cvar)
            marginal_cvar = np.divide(
                component_cvar,
                weights,
                out=np.zeros_like(component_cvar),
                where=np.abs(weights) > 1e-15,
            )
            method = "Numerical Euler (scenario tail expectations)"

        # Incremental VaR of a holding is VaR(with) - VaR(without): positive
        # when removing the position would reduce portfolio risk.
        incremental_var = np.zeros_like(weights)
        for index in range(weights.size):
            incremental_var[index] = -self.incremental_var_for_adjustment(index, -weights[index])

        return RiskDecomposition(
            tickers=model.assets,
            weights=weights.copy(),
            marginal_var=np.asarray(marginal_var, dtype=float),
            component_var=np.asarray(component_var, dtype=float),
            incremental_var=incremental_var,
            component_cvar=np.asarray(component_cvar, dtype=float),
            marginal_cvar=np.asarray(marginal_cvar, dtype=float),
            total_var=total_var,
            total_cvar=total_cvar,
            method=method,
            model_name=model.name,
        )


def decompose_model(model: RiskModel) -> RiskDecomposition:
    """Convenience wrapper around :class:`EulerRiskDecomposer`.

    Args:
        model: A fitted risk model.

    Returns:
        The risk attribution.
    """
    return EulerRiskDecomposer(model).decompose()


def align_weights(tickers: Sequence[str], weights: dict[str, float] | Sequence[float]) -> np.ndarray:
    """Build a weight vector aligned with ``tickers``.

    Args:
        tickers: Asset order for the output vector.
        weights: Either a mapping ``ticker -> weight`` (missing assets get zero)
            or a sequence already aligned with ``tickers``.

    Returns:
        Float weight vector.

    Raises:
        RiskModelError: If a sequence length mismatches or all weights are zero.
    """
    if isinstance(weights, dict):
        vector = np.array([float(weights.get(ticker, 0.0)) for ticker in tickers], dtype=float)
    else:
        vector = np.asarray(list(weights), dtype=float).ravel()
        if vector.size != len(tickers):
            raise RiskModelError(
                f"Expected {len(tickers)} weights for {list(tickers)}, received {vector.size}."
            )
    if np.allclose(vector, 0.0):
        raise RiskModelError("Portfolio weights sum to zero; nothing to decompose.")
    return vector


def format_decomposition_table(decomposition: RiskDecomposition, portfolio_value: float = 1.0) -> pd.DataFrame:
    """Return a currency-annotated attribution table for reporting.

    Args:
        decomposition: Attribution result.
        portfolio_value: Notional used to convert fractions to currency.

    Returns:
        DataFrame indexed by asset with percentage and currency columns.
    """
    frame = decomposition.to_frame()
    frame["weight_pct"] = 100.0 * frame["weight"]
    frame["marginal_var_pct"] = 100.0 * frame["marginal_var"]
    frame["component_var_pct"] = 100.0 * frame["component_var"]
    frame["component_var_amount"] = frame["component_var"] * portfolio_value
    frame["component_share_pct"] = 100.0 * frame["component_share"]
    frame["incremental_var_amount"] = frame["incremental_var"] * portfolio_value
    frame["component_cvar_pct"] = 100.0 * frame["component_cvar"]
    return frame
