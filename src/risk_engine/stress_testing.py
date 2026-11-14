"""Stress testing: historical crisis replay and hypothetical covariance shocks.

Two families of scenarios are supported:

* **Historical replay** - the actual asset returns inside a crisis window are
  applied to today's weights. The portfolio is treated as buy-and-hold over the
  window, so each asset's P&L contribution is
  ``V * w_i * (prod(1 + r_i) - 1)`` and contributions sum exactly to the total.
* **Hypothetical shock** - a deterministic mark-to-market shock (equities -15%
  by default) combined with a covariance breakdown: volatilities scaled up,
  correlations forced towards a common level. The engine reports the P&L of the
  shock and the resulting VaR/CVaR increase relative to the base covariance.

Risk numbers are positive loss fractions; P&L numbers are signed currency
amounts where negative values are losses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .data_loader import MarketData
from .decomposition import align_weights
from .risk_models import RiskModelError, ensure_positive_semidefinite, portfolio_variance, sample_covariance

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalStressScenario:
    """Definition of a historical crisis window.

    Attributes:
        name: Human-readable scenario name.
        start: First date of the replay window (inclusive).
        end: Last date of the replay window (inclusive).
        description: Economic narrative of the episode.
    """

    name: str
    start: str
    end: str
    description: str


DEFAULT_HISTORICAL_SCENARIOS: tuple[HistoricalStressScenario, ...] = (
    HistoricalStressScenario(
        name="2008 Global Financial Crisis",
        start="2008-09-01",
        end="2008-11-30",
        description="Lehman collapse, funding freeze and cross-asset deleveraging.",
    ),
    HistoricalStressScenario(
        name="2020 COVID-19 Liquidity Shock",
        start="2020-02-19",
        end="2020-03-23",
        description="Fastest bear market on record; dash for cash across asset classes.",
    ),
    HistoricalStressScenario(
        name="2022 Inflation & Rate Shock",
        start="2022-01-03",
        end="2022-10-31",
        description="Sticky inflation, aggressive Fed tightening; stocks and bonds fell together.",
    ),
)


@dataclass(frozen=True)
class HistoricalStressResult:
    """Outcome of replaying one historical window on the current portfolio.

    Attributes:
        name: Scenario name.
        start: First replayed return date.
        end: Last replayed return date.
        description: Scenario narrative.
        trading_days: Number of return observations replayed.
        portfolio_value: Starting notional.
        portfolio_pnl: Signed P&L of the replay.
        portfolio_return: Signed portfolio return over the window.
        worst_day_pnl: Worst single-day P&L.
        worst_day_date: Date of the worst single day.
        max_drawdown: Maximum peak-to-trough drawdown of the equity path
            (negative fraction).
        asset_returns: Cumulative simple return per asset over the window.
        asset_pnl: P&L contribution per asset (sums to ``portfolio_pnl``).
        equity_path: Portfolio value path under buy-and-hold.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    description: str
    trading_days: int
    portfolio_value: float
    portfolio_pnl: float
    portfolio_return: float
    worst_day_pnl: float
    worst_day_date: pd.Timestamp
    max_drawdown: float
    asset_returns: pd.Series
    asset_pnl: pd.Series
    equity_path: pd.Series

    def to_frame(self) -> pd.DataFrame:
        """Return per-asset contribution rows."""
        frame = pd.DataFrame(
            {
                "asset_return": self.asset_returns,
                "pnl": self.asset_pnl,
            }
        )
        total = float(self.asset_pnl.sum())
        frame["share"] = self.asset_pnl / total if abs(total) > 1e-12 else 0.0
        return frame


@dataclass(frozen=True)
class HypotheticalShock:
    """Parameters of a hypothetical covariance-breakdown shock.

    Attributes:
        name: Scenario name.
        equity_shock: Deterministic return applied to equity assets.
        equity_assets: Assets treated as equities for the deterministic shock.
        vol_multiplier: Multiplier applied to every asset volatility.
        correlation_target: Common off-diagonal correlation after the shock.
        correlation_blend: Weight of the stressed correlation matrix
            (``1.0`` forces all pairwise correlations to the target).
    """

    name: str = "Hypothetical Covariance Breakdown"
    equity_shock: float = -0.15
    equity_assets: tuple[str, ...] = ("SPY", "QQQ")
    vol_multiplier: float = 2.0
    correlation_target: float = 0.85
    correlation_blend: float = 1.0
    description: str = (
        "Equity shock with a volatility spike and diversification failure: "
        "cross-asset correlations converge towards a single stressed factor."
    )


@dataclass(frozen=True)
class HypotheticalStressResult:
    """Outcome of the hypothetical covariance-breakdown shock.

    Attributes:
        name: Scenario name.
        description: Scenario narrative.
        shock: The parameter set applied.
        applied_assets: Assets that received the deterministic equity shock.
        portfolio_pnl: Signed P&L of the deterministic shock.
        portfolio_return: Signed portfolio return of the shock.
        asset_pnl: P&L per asset from the deterministic shock.
        base_var: Base 1-day VaR at the model confidence (loss fraction).
        shocked_var: VaR under the stressed covariance (loss fraction).
        var_increase: Absolute VaR increase.
        base_cvar: Base CVaR.
        shocked_cvar: Stressed CVaR.
        base_vol_annual: Base annualised portfolio volatility.
        shocked_vol_annual: Stressed annualised portfolio volatility.
        base_average_correlation: Mean off-diagonal correlation before the shock.
        shocked_average_correlation: Mean off-diagonal correlation after the shock.
        base_diversification: ``1 - sigma_p / sum(|w_i| sigma_i)`` before the shock.
        shocked_diversification: The same ratio after the shock.
    """

    name: str
    description: str
    shock: HypotheticalShock
    applied_assets: tuple[str, ...]
    portfolio_pnl: float
    portfolio_return: float
    asset_pnl: pd.Series
    base_var: float
    shocked_var: float
    var_increase: float
    base_cvar: float
    shocked_cvar: float
    base_vol_annual: float
    shocked_vol_annual: float
    base_average_correlation: float
    shocked_average_correlation: float
    base_diversification: float
    shocked_diversification: float


def _average_off_diagonal_correlation(covariance: np.ndarray) -> float:
    """Mean of the off-diagonal correlation entries (0 for one asset)."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape[0] < 2:
        return 0.0
    volatility = np.sqrt(np.clip(np.diag(matrix), 1e-18, None))
    correlation = matrix / np.outer(volatility, volatility)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(correlation[mask].mean())


def _diversification_ratio(weights: np.ndarray, covariance: np.ndarray) -> float:
    """``1 - portfolio vol / weighted average vol`` (0 = no diversification)."""
    sigma_p = float(np.sqrt(portfolio_variance(weights, covariance)))
    volatility = np.sqrt(np.clip(np.diag(covariance), 1e-18, None))
    weighted = float(np.sum(np.abs(weights) * volatility))
    if weighted <= 1e-18:
        return 0.0
    return float(1.0 - sigma_p / weighted)


def stressed_covariance(
    covariance: np.ndarray,
    *,
    vol_multiplier: float = 2.0,
    correlation_target: float = 0.85,
    correlation_blend: float = 1.0,
) -> np.ndarray:
    """Build a stressed covariance matrix with inflated vols and correlations.

    The shock is applied in two steps:

    1. Correlation matrix ``R`` is blended towards a matrix whose off-diagonal
       entries all equal ``correlation_target``.
    2. Every volatility is multiplied by ``vol_multiplier``, giving
       ``Sigma* = D R* D``. The result is projected onto the PSD cone.

    Args:
        covariance: Base covariance matrix.
        vol_multiplier: Volatility multiplier (>= 1).
        correlation_target: Target off-diagonal correlation.
        correlation_blend: Blend weight in ``[0, 1]``.

    Returns:
        Stressed covariance matrix.

    Raises:
        RiskModelError: For invalid parameters.
    """
    if vol_multiplier <= 0:
        raise RiskModelError("vol_multiplier must be positive.")
    if not -1.0 <= correlation_target <= 1.0:
        raise RiskModelError("correlation_target must lie in [-1, 1].")
    if not 0.0 <= correlation_blend <= 1.0:
        raise RiskModelError("correlation_blend must lie in [0, 1].")

    matrix = np.asarray(covariance, dtype=float)
    volatility = np.sqrt(np.clip(np.diag(matrix), 1e-18, None))
    correlation = matrix / np.outer(volatility, volatility)
    np.fill_diagonal(correlation, 1.0)
    target = np.full_like(correlation, correlation_target)
    np.fill_diagonal(target, 1.0)
    blended = (1.0 - correlation_blend) * correlation + correlation_blend * target
    np.fill_diagonal(blended, 1.0)
    stressed = blended * np.outer(volatility * vol_multiplier, volatility * vol_multiplier)
    return ensure_positive_semidefinite(stressed)


class StressTester:
    """Run historical and hypothetical stress scenarios on a portfolio.

    Example:
        >>> tester = StressTester(market_data, {"SPY": 0.4, "TLT": 0.6}, 1_000_000)
        >>> results = tester.run_all_historical()
        >>> tester.run_hypothetical().var_increase
        0.0123...
    """

    def __init__(
        self,
        market_data: MarketData,
        weights: Mapping[str, float] | Sequence[float],
        portfolio_value: float = 1_000_000.0,
        confidence: float = 0.99,
        horizon_days: int = 1,
    ) -> None:
        """Initialise the stress tester.

        Args:
            market_data: Cleaned multi-asset price panel.
            weights: Mapping ``ticker -> weight`` or a sequence aligned with
                ``market_data.tickers``.
            portfolio_value: Current portfolio notional.
            confidence: Confidence level for the stressed VaR/CVaR comparison.
            horizon_days: Horizon in trading days for the VaR comparison.

        Raises:
            RiskModelError: For invalid weights or parameters.
        """
        if portfolio_value <= 0:
            raise RiskModelError("portfolio_value must be positive.")
        if not 0.5 < confidence < 1.0:
            raise RiskModelError("confidence must lie between 0.5 and 1.0.")
        self.market_data = market_data
        self.tickers = market_data.tickers
        self.weights = align_weights(self.tickers, weights)
        self.portfolio_value = float(portfolio_value)
        self.confidence = float(confidence)
        self.horizon_days = int(horizon_days)
        self.returns = market_data.simple_returns

    # -- historical -----------------------------------------------------
    def _window_returns(self, scenario: HistoricalStressScenario) -> pd.DataFrame:
        window = self.market_data.slice(scenario.start, scenario.end)
        frame = window.simple_returns
        if frame.empty:
            raise RiskModelError(f"No returns available for {scenario.name}.")
        return frame.loc[:, list(self.tickers)]

    def run_historical(self, scenario: HistoricalStressScenario) -> HistoricalStressResult:
        """Replay one historical crisis window on the current weights.

        Args:
            scenario: Window definition.

        Returns:
            A :class:`HistoricalStressResult`.

        Raises:
            RiskModelError: If the window is unavailable in the data.
        """
        returns = self._window_returns(scenario)
        weights = self.weights
        growth = (1.0 + returns).cumprod()
        asset_returns = growth.iloc[-1] - 1.0
        asset_pnl = self.portfolio_value * weights * asset_returns
        portfolio_pnl = float(asset_pnl.sum())
        equity_path = self.portfolio_value * (1.0 + ((growth - 1.0) @ weights))
        daily_pnl = equity_path.diff()
        if not daily_pnl.empty:
            daily_pnl.iloc[0] = equity_path.iloc[0] - self.portfolio_value
        running_peak = equity_path.cummax()
        drawdown = equity_path / running_peak - 1.0
        worst_date = pd.Timestamp(daily_pnl.idxmin()) if not daily_pnl.empty else pd.Timestamp(returns.index[-1])
        return HistoricalStressResult(
            name=scenario.name,
            start=pd.Timestamp(returns.index[0]),
            end=pd.Timestamp(returns.index[-1]),
            description=scenario.description,
            trading_days=int(len(returns)),
            portfolio_value=self.portfolio_value,
            portfolio_pnl=portfolio_pnl,
            portfolio_return=portfolio_pnl / self.portfolio_value,
            worst_day_pnl=float(daily_pnl.min()) if not daily_pnl.empty else 0.0,
            worst_day_date=worst_date,
            max_drawdown=float(drawdown.min()) if not drawdown.empty else 0.0,
            asset_returns=asset_returns.rename("asset_return"),
            asset_pnl=asset_pnl.rename("pnl"),
            equity_path=equity_path.rename("equity"),
        )

    def run_all_historical(
        self,
        scenarios: Sequence[HistoricalStressScenario] = DEFAULT_HISTORICAL_SCENARIOS,
        *,
        skip_unavailable: bool = True,
    ) -> tuple[HistoricalStressResult, ...]:
        """Replay every historical scenario, optionally skipping missing windows.

        Args:
            scenarios: Scenario definitions.
            skip_unavailable: If ``True``, log and skip windows outside the
                data range instead of raising.

        Returns:
            Tuple of results in scenario order.

        Raises:
            RiskModelError: If no scenario could be evaluated, or if a window is
                unavailable and ``skip_unavailable`` is ``False``.
        """
        results: list[HistoricalStressResult] = []
        for scenario in scenarios:
            try:
                results.append(self.run_historical(scenario))
            except Exception as exc:
                if not skip_unavailable:
                    raise
                LOGGER.warning("Skipping stress scenario %s: %s", scenario.name, exc)
        if not results:
            raise RiskModelError("No historical stress scenario could be evaluated with the available data.")
        return tuple(results)

    # -- hypothetical ---------------------------------------------------
    def _shock_assets(self, shock: HypotheticalShock) -> tuple[str, ...]:
        """Resolve which assets receive the deterministic equity shock."""
        available = tuple(t for t in shock.equity_assets if t in self.tickers)
        if available:
            return available
        volatility = np.sqrt(np.clip(np.diag(sample_covariance(self.returns.to_numpy(dtype=float))), 1e-18, None))
        threshold = np.quantile(volatility, 0.66)
        fallback = tuple(
            ticker for ticker, vol in zip(self.tickers, volatility) if vol >= threshold
        )
        LOGGER.warning(
            "None of the configured equity assets %s are present; applying the shock to the "
            "highest-volatility assets %s.",
            shock.equity_assets,
            fallback,
        )
        return fallback

    def run_hypothetical(self, shock: HypotheticalShock | None = None) -> HypotheticalStressResult:
        """Apply the deterministic shock and the covariance breakdown.

        Args:
            shock: Shock parameters (defaults to :class:`HypotheticalShock`).

        Returns:
            A :class:`HypotheticalStressResult` with P&L and VaR deltas.

        Raises:
            RiskModelError: If the estimation sample is too short.
        """
        parameters = shock or HypotheticalShock()
        returns = self.returns.to_numpy(dtype=float)
        if returns.shape[0] < max(30, returns.shape[1] + 2):
            raise RiskModelError("Insufficient history for the hypothetical shock.")
        weights = self.weights
        mu = returns.mean(axis=0)
        covariance = sample_covariance(returns)
        shock_vector = np.zeros_like(weights)
        applied = self._shock_assets(parameters)
        for ticker in applied:
            shock_vector[self.tickers.index(ticker)] = parameters.equity_shock

        asset_pnl = self.portfolio_value * weights * shock_vector
        portfolio_pnl = float(asset_pnl.sum())
        shocked_covariance = stressed_covariance(
            covariance,
            vol_multiplier=parameters.vol_multiplier,
            correlation_target=parameters.correlation_target,
            correlation_blend=parameters.correlation_blend,
        )

        horizon = float(self.horizon_days)
        z = float(stats.norm.ppf(1.0 - self.confidence))
        es_factor = float(stats.norm.pdf(z) / (1.0 - self.confidence))

        def measure(sigma_matrix: np.ndarray) -> tuple[float, float]:
            mean_p = float(weights @ mu) * horizon
            sigma_p = float(np.sqrt(portfolio_variance(weights, sigma_matrix))) * np.sqrt(horizon)
            var = -(mean_p + z * sigma_p)
            cvar = -mean_p + es_factor * sigma_p
            return float(var), float(cvar)

        base_var, base_cvar = measure(covariance)
        shocked_var, shocked_cvar = measure(shocked_covariance)

        def annual_vol(sigma_matrix: np.ndarray) -> float:
            return float(np.sqrt(portfolio_variance(weights, sigma_matrix)) * np.sqrt(252.0))

        return HypotheticalStressResult(
            name=parameters.name,
            description=parameters.description,
            shock=parameters,
            applied_assets=applied,
            portfolio_pnl=portfolio_pnl,
            portfolio_return=portfolio_pnl / self.portfolio_value,
            asset_pnl=pd.Series(asset_pnl, index=list(self.tickers), name="pnl"),
            base_var=base_var,
            shocked_var=shocked_var,
            var_increase=shocked_var - base_var,
            base_cvar=base_cvar,
            shocked_cvar=shocked_cvar,
            base_vol_annual=annual_vol(covariance),
            shocked_vol_annual=annual_vol(shocked_covariance),
            base_average_correlation=_average_off_diagonal_correlation(covariance),
            shocked_average_correlation=_average_off_diagonal_correlation(shocked_covariance),
            base_diversification=_diversification_ratio(weights, covariance),
            shocked_diversification=_diversification_ratio(weights, shocked_covariance),
        )


def stress_summary_table(results: Sequence[HistoricalStressResult]) -> pd.DataFrame:
    """Build a reporting table from historical stress results.

    Args:
        results: Stress results in display order.

    Returns:
        One row per scenario with P&L, return, worst day and drawdown.
    """
    rows = []
    for result in results:
        rows.append(
            {
                "scenario": result.name,
                "window": f"{result.start.date()} to {result.end.date()}",
                "days": result.trading_days,
                "pnl": result.portfolio_pnl,
                "return_pct": 100.0 * result.portfolio_return,
                "worst_day_pnl": result.worst_day_pnl,
                "max_drawdown_pct": 100.0 * result.max_drawdown,
            }
        )
    return pd.DataFrame(rows)
