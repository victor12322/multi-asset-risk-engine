"""Rolling VaR backtesting with Kupiec and Christoffersen coverage tests.

The backtest is strictly out-of-sample: for each test date ``t`` the model is
estimated on the previous ``window`` observations and its VaR forecast is
compared with the realised P&L at ``t``. A breach (exception) occurs when the
loss strictly exceeds the forecast VaR.

Statistical tests
-----------------
* **Kupiec (1995) Proportion of Failures**

  .. math::
      LR_{POF} = -2 \\ln \\frac{(1-p)^{T-x} p^{x}}
      {\\big(1-\\hat p\\big)^{T-x} \\hat p^{x}}, \\qquad
      \\hat p = x/T, \\quad p = 1-\\alpha

* **Christoffersen (1998) independence and conditional coverage**

  .. math::
      LR_{IND} = -2 \\ln \\frac{(1-\\pi)^{n_{00}+n_{10}} \\pi^{n_{01}+n_{11}}}
      {(1-\\pi_{01})^{n_{00}} \\pi_{01}^{n_{01}}
      (1-\\pi_{11})^{n_{10}} \\pi_{11}^{n_{11}}}, \\qquad
      LR_{CC} = LR_{POF} + LR_{IND} \\sim \\chi^2(2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from .risk_models import (
    FilteredEWMAStudentTRiskModel,
    InsufficientDataError,
    MonteCarloStudentTRiskModel,
    ParametricNormalRiskModel,
    RiskModel,
    RiskModelError,
    _as_weights,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KupiecResult:
    """Kupiec Proportion-of-Failures test outcome.

    Attributes:
        confidence: VaR confidence level.
        n_observations: Number of out-of-sample forecasts.
        exceptions: Number of VaR breaches.
        expected_exceptions: Expected breaches under correct coverage.
        failure_rate: Observed breach rate ``x / T``.
        lr_pof: Likelihood-ratio statistic (chi-square, 1 degree of freedom).
        p_value: Survival-function p-value.
        passed: ``True`` when the null of correct coverage is not rejected.
        significance: Test size.
    """

    confidence: float
    n_observations: int
    exceptions: int
    expected_exceptions: float
    failure_rate: float
    lr_pof: float
    p_value: float
    passed: bool
    significance: float

    def as_row(self) -> dict[str, float | int | bool]:
        """Flatten the result for tabular reporting."""
        return {
            "n_observations": self.n_observations,
            "exceptions": self.exceptions,
            "expected_exceptions": self.expected_exceptions,
            "failure_rate": self.failure_rate,
            "lr_pof": self.lr_pof,
            "p_value": self.p_value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ChristoffersenResult:
    """Christoffersen independence and conditional-coverage test outcome.

    Attributes:
        n00: No breach after no breach.
        n01: Breach after no breach.
        n10: No breach after breach.
        n11: Breach after breach.
        lr_independence: ``LR_IND`` statistic (chi-square, 1 dof).
        p_independence: p-value of ``LR_IND``.
        lr_conditional_coverage: ``LR_CC = LR_POF + LR_IND`` (chi-square, 2 dof).
        p_conditional_coverage: p-value of ``LR_CC``.
        passed_independence: Independence not rejected at ``significance``.
        passed_conditional_coverage: Conditional coverage not rejected.
        independence_defined: ``False`` when the transition matrix has an empty
            row and ``LR_IND`` is undefined; in that case ``LR_IND`` is set to
            zero and the conditional-coverage test reduces to Kupiec.
        significance: Test size.
    """

    n00: int
    n01: int
    n10: int
    n11: int
    lr_independence: float
    p_independence: float
    lr_conditional_coverage: float
    p_conditional_coverage: float
    passed_independence: bool
    passed_conditional_coverage: bool
    independence_defined: bool
    significance: float

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the 2x2 transition counts as ``[[n00, n01], [n10, n11]]``."""
        return np.array([[self.n00, self.n01], [self.n10, self.n11]], dtype=int)

    def as_row(self) -> dict[str, float | int | bool]:
        """Flatten the result for tabular reporting."""
        return {
            "n00": self.n00,
            "n01": self.n01,
            "n10": self.n10,
            "n11": self.n11,
            "lr_independence": self.lr_independence,
            "p_independence": self.p_independence,
            "lr_conditional_coverage": self.lr_conditional_coverage,
            "p_conditional_coverage": self.p_conditional_coverage,
            "passed_independence": self.passed_independence,
            "passed_conditional_coverage": self.passed_conditional_coverage,
            "independence_defined": self.independence_defined,
            "significance": self.significance,
        }


def kupiec_pof_test(
    exceptions: int,
    n_observations: int,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> KupiecResult:
    """Kupiec Proportion-of-Failures likelihood-ratio test.

    Args:
        exceptions: Number of VaR breaches.
        n_observations: Number of forecasts.
        confidence: VaR confidence level ``alpha``.
        significance: Test size used for pass/fail verdicts.

    Returns:
        A :class:`KupiecResult`.

    Raises:
        RiskModelError: For invalid counts or confidence levels.
    """
    if not 0.5 < confidence < 1.0:
        raise RiskModelError("confidence must lie between 0.5 and 1.0.")
    if int(n_observations) <= 0:
        raise RiskModelError("n_observations must be positive.")
    if not 0 <= int(exceptions) <= int(n_observations):
        raise RiskModelError("exceptions must lie between 0 and n_observations.")
    total = int(n_observations)
    failures = int(exceptions)
    target = 1.0 - float(confidence)
    observed = failures / total

    null_loglik = xlogy(total - failures, 1.0 - target) + xlogy(failures, target)
    alternative_loglik = xlogy(total - failures, 1.0 - observed) + xlogy(failures, observed)
    statistic = max(0.0, float(-2.0 * (null_loglik - alternative_loglik)))
    p_value = float(stats.chi2.sf(statistic, df=1))
    return KupiecResult(
        confidence=float(confidence),
        n_observations=total,
        exceptions=failures,
        expected_exceptions=total * target,
        failure_rate=observed,
        lr_pof=statistic,
        p_value=p_value,
        passed=bool(p_value >= significance),
        significance=float(significance),
    )


def christoffersen_test(
    breaches: Sequence[bool] | Sequence[int] | np.ndarray,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> ChristoffersenResult:
    """Christoffersen independence and conditional-coverage tests.

    Args:
        breaches: Consecutive breach indicators (one per forecast date, in
            chronological order).
        confidence: VaR confidence level used for ``LR_POF``.
        significance: Test size used for pass/fail verdicts.

    Returns:
        A :class:`ChristoffersenResult` with ``LR_IND`` and
        ``LR_CC = LR_POF + LR_IND``.

    Raises:
        RiskModelError: If fewer than two observations are supplied.
    """
    flags = np.asarray([bool(flag) for flag in breaches], dtype=bool)
    if flags.size < 2:
        raise RiskModelError("At least two breach observations are required.")
    previous = flags[:-1]
    current = flags[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))
    total = int(flags.size)
    exceptions = int(flags.sum())
    kupiec = kupiec_pof_test(exceptions, total, confidence=confidence, significance=significance)

    row_zero = n00 + n01
    row_one = n10 + n11
    independence_defined = row_zero > 0 and row_one > 0 and (n01 + n11) > 0
    if independence_defined:
        pi01 = n01 / row_zero
        pi11 = n11 / row_one
        pi = (n01 + n11) / (row_zero + row_one)
        unrestricted = (
            xlogy(n00, 1.0 - pi01)
            + xlogy(n01, pi01)
            + xlogy(n10, 1.0 - pi11)
            + xlogy(n11, pi11)
        )
        restricted = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
        lr_independence = max(0.0, float(-2.0 * (restricted - unrestricted)))
    else:
        lr_independence = 0.0

    p_independence = float(stats.chi2.sf(lr_independence, df=1)) if independence_defined else 1.0
    lr_cc = float(kupiec.lr_pof + lr_independence)
    p_cc = float(stats.chi2.sf(lr_cc, df=2))
    return ChristoffersenResult(
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        lr_independence=lr_independence,
        p_independence=p_independence,
        lr_conditional_coverage=lr_cc,
        p_conditional_coverage=p_cc,
        passed_independence=bool(p_independence >= significance),
        passed_conditional_coverage=bool(p_cc >= significance),
        independence_defined=bool(independence_defined),
        significance=float(significance),
    )


@dataclass(frozen=True)
class BacktestResult:
    """Rolling out-of-sample VaR backtest for a single model.

    Attributes:
        model_name: Model label.
        confidence: VaR confidence level.
        window: Estimation window length in observations.
        horizon_days: VaR horizon in trading days.
        pnl: Realised losses (positive = loss) for each test date.
        var: Forecast VaR (positive loss fraction) for each test date.
        breaches: Boolean breach indicators aligned with ``pnl``.
        kupiec: Kupiec POF result.
        christoffersen: Christoffersen independence/CC result.
        significance: Test size used for verdicts.
    """

    model_name: str
    confidence: float
    window: int
    horizon_days: int
    pnl: pd.Series
    var: pd.Series
    breaches: pd.Series
    kupiec: KupiecResult
    christoffersen: ChristoffersenResult
    significance: float = 0.05

    @property
    def n_observations(self) -> int:
        """Number of out-of-sample forecasts."""
        return int(self.pnl.size)

    @property
    def exceptions(self) -> int:
        """Number of VaR breaches."""
        return int(self.breaches.sum())

    @property
    def breach_rate(self) -> float:
        """Observed breach rate."""
        return float(self.exceptions / self.n_observations) if self.n_observations else float("nan")

    @property
    def breach_dates(self) -> pd.DatetimeIndex:
        """Dates on which VaR was breached."""
        return pd.DatetimeIndex(self.breaches.index[self.breaches.to_numpy(dtype=bool)])

    def to_frame(self) -> pd.DataFrame:
        """Return the per-date backtest series."""
        return pd.DataFrame({"pnl": self.pnl, "var": self.var, "breach": self.breaches})

    def summary_row(self) -> dict[str, float | str]:
        """Flatten the result into a single reporting row."""
        return {
            "model": self.model_name,
            "observations": self.n_observations,
            "exceptions": self.exceptions,
            "expected": self.kupiec.expected_exceptions,
            "breach_rate_pct": 100.0 * self.breach_rate,
            "kupiec_lr": self.kupiec.lr_pof,
            "kupiec_p": self.kupiec.p_value,
            "kupiec_verdict": "PASS" if self.kupiec.passed else "FAIL",
            "ind_lr": self.christoffersen.lr_independence,
            "ind_p": self.christoffersen.p_independence,
            "ind_verdict": "PASS" if self.christoffersen.passed_independence else "FAIL",
            "cc_lr": self.christoffersen.lr_conditional_coverage,
            "cc_p": self.christoffersen.p_conditional_coverage,
            "cc_verdict": "PASS" if self.christoffersen.passed_conditional_coverage else "FAIL",
        }


def rolling_var_backtest(
    returns: pd.DataFrame | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    model_factory: Callable[[], RiskModel],
    *,
    window: int = 250,
    confidence: float = 0.99,
    horizon_days: int = 1,
    refit_every: int = 1,
    significance: float = 0.05,
) -> BacktestResult:
    """Run a rolling out-of-sample VaR backtest.

    Args:
        returns: Daily asset returns (``T`` rows).
        weights: Portfolio weights.
        model_factory: Zero-argument callable returning an **unfitted** model.
        window: Estimation window length in observations.
        confidence: Confidence level (must match the models produced by the
            factory).
        horizon_days: VaR horizon in trading days.
        refit_every: Re-estimate the model every ``refit_every`` test dates.
        significance: Test size for verdicts.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        InsufficientDataError: If there is not enough history for a meaningful
            out-of-sample test.
        RiskModelError: For invalid parameters.
    """
    if isinstance(returns, pd.DataFrame):
        frame = returns.astype(float)
        array = frame.to_numpy(dtype=float)
        index = frame.index
    else:
        array = np.asarray(returns, dtype=float)
        if array.ndim != 2:
            raise RiskModelError("returns must be a (T, N) array.")
        index = pd.RangeIndex(array.shape[0])
        frame = pd.DataFrame(array, index=index)
    if window < 20:
        raise RiskModelError("window must be at least 20 observations.")
    if refit_every < 1:
        raise RiskModelError("refit_every must be at least 1.")
    if array.shape[0] <= window + 20:
        raise InsufficientDataError(
            f"Backtest needs more than {window + 20} observations, received {array.shape[0]}."
        )

    weight_vector = _as_weights(weights, array.shape[1])
    portfolio_returns = array @ weight_vector
    n_observations = array.shape[0]
    forecast_var = np.full(n_observations, np.nan, dtype=float)
    model: RiskModel | None = None
    base_seed: int | None = None
    refits = 0

    for t in range(window, n_observations):
        if model is None or (t - window) % refit_every == 0:
            if model is None:
                model = model_factory()
                base_seed = getattr(model, "seed", None)
            elif base_seed is not None and hasattr(model, "seed"):
                # Fresh, deterministic draws for every re-estimation.
                model.seed = int(base_seed) + refits
            model.fit(frame.iloc[t - window : t], weight_vector)
            refits += 1
        assert model is not None
        forecast_var[t] = model.var_for_weights(weight_vector)

    test_index = index[window:]
    pnl = pd.Series(-portfolio_returns[window:], index=test_index, name="pnl")
    var_series = pd.Series(forecast_var[window:], index=test_index, name="var")
    if var_series.isna().any():  # pragma: no cover - defensive
        raise RiskModelError("Rolling backtest produced undefined VaR forecasts.")
    breaches = (pnl > var_series).rename("breach")
    kupiec = kupiec_pof_test(int(breaches.sum()), int(breaches.size), confidence, significance)
    christoffersen = christoffersen_test(breaches.to_numpy(dtype=bool), confidence, significance)
    return BacktestResult(
        model_name=model.name if model is not None else "unknown",
        confidence=confidence,
        window=int(window),
        horizon_days=int(horizon_days),
        pnl=pnl,
        var=var_series,
        breaches=breaches,
        kupiec=kupiec,
        christoffersen=christoffersen,
        significance=significance,
    )


def run_standard_backtests(
    returns: pd.DataFrame,
    weights: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.99,
    window: int = 250,
    mc_paths: int = 10_000,
    seed: int = 11,
    refit_every: int = 5,
    covariance_method: str = "sample",
    student_t_dof: float = 4.0,
    filtered_estimate_dof: bool = True,
    significance: float = 0.05,
    models: Sequence[str] = ("normal", "mc_student_t", "filtered_ewma_t"),
) -> dict[str, BacktestResult]:
    """Backtest the static Normal, static Student-t and filtered EWMA models.

    Args:
        returns: Daily asset returns.
        weights: Portfolio weights.
        confidence: Confidence level (99% by default).
        window: Rolling estimation window.
        mc_paths: Monte Carlo paths per re-estimation.
        seed: Base Monte Carlo seed.
        refit_every: Re-estimation frequency in test dates.
        covariance_method: ``"sample"`` or ``"ewma"``.
        student_t_dof: Fixed degrees of freedom for the Student-t models.
        filtered_estimate_dof: Estimate ``nu`` from EWMA-standardised residuals.
        significance: Test size for verdicts.
        models: Model keys to run from
            ``{"normal", "mc_student_t", "filtered_ewma_t"}``.

    Returns:
        Mapping ``model_key -> BacktestResult`` in the requested order.

    Raises:
        RiskModelError: For an unsupported model key.
    """
    factories: dict[str, Callable[[], RiskModel]] = {
        "normal": lambda: ParametricNormalRiskModel(
            confidence=confidence, covariance_method=covariance_method
        ),
        "mc_student_t": lambda: MonteCarloStudentTRiskModel(
            confidence=confidence,
            n_paths=mc_paths,
            seed=seed,
            degrees_of_freedom=student_t_dof,
            covariance_method=covariance_method,
        ),
        "filtered_ewma_t": lambda: FilteredEWMAStudentTRiskModel(
            confidence=confidence,
            decay=0.94,
            degrees_of_freedom=student_t_dof,
            estimate_dof=filtered_estimate_dof,
        ),
    }
    results: dict[str, BacktestResult] = {}
    for key in models:
        if key not in factories:
            raise RiskModelError(f"Unsupported backtest model '{key}'. Choose from {sorted(factories)}.")
        results[key] = rolling_var_backtest(
            returns,
            weights,
            factories[key],
            window=window,
            confidence=confidence,
            refit_every=refit_every,
            significance=significance,
        )
        LOGGER.info(
            "Backtest %s complete: %d exceptions of %d forecasts (LR_POF=%.3f, p=%.3f).",
            results[key].model_name,
            results[key].exceptions,
            results[key].n_observations,
            results[key].kupiec.lr_pof,
            results[key].kupiec.p_value,
        )
    return results


def backtest_summary_table(results: Mapping[str, BacktestResult]) -> pd.DataFrame:
    """Build a consolidated backtest verdict table.

    Args:
        results: Model key -> backtest result.

    Returns:
        DataFrame with one row per model.
    """
    return pd.DataFrame([result.summary_row() for result in results.values()])
