"""Multi-Asset Portfolio Risk Engine & Regulatory Backtesting Framework.

A production-grade Python library for multi-asset portfolio risk measurement,
Euler risk attribution, crisis stress testing and regulatory VaR backtesting.

Example:
    >>> from risk_engine.data_loader import load_market_data
    >>> from risk_engine.decomposition import DEFAULT_WEIGHTS, align_weights
    >>> from risk_engine.risk_models import build_risk_models
    >>> market = load_market_data(offline=True)
    >>> weights = align_weights(market.tickers, DEFAULT_WEIGHTS)
    >>> models = build_risk_models(market.simple_returns, weights, confidence=0.99)
    >>> round(models["normal"].var(), 4) >= 0
    True
"""

from __future__ import annotations

from .backtesting import (
    BacktestResult,
    ChristoffersenResult,
    KupiecResult,
    backtest_summary_table,
    christoffersen_test,
    kupiec_pof_test,
    rolling_var_backtest,
    run_standard_backtests,
)
from .data_loader import (
    DEFAULT_START,
    DEFAULT_TICKERS,
    DataLoadError,
    DataQualityReport,
    MarketData,
    generate_synthetic_market_data,
    load_csv_prices,
    load_market_data,
)
from .decomposition import (
    DEFAULT_WEIGHTS,
    EULER_TOLERANCE,
    EulerRiskDecomposer,
    RiskDecomposition,
    align_weights,
    analytic_normal_euler,
    empirical_component_cvar,
)
from .risk_models import (
    MODEL_KEYS,
    MODEL_LABELS,
    FilteredEWMAStudentTRiskModel,
    HistoricalSimulationRiskModel,
    InsufficientDataError,
    MonteCarloNormalRiskModel,
    MonteCarloStudentTRiskModel,
    ParametricNormalRiskModel,
    RiskMeasureResult,
    RiskModel,
    RiskModelError,
    build_risk_models,
    estimate_degrees_of_freedom,
    evaluate_models,
    ewma_covariance,
    ewma_volatility,
    sample_covariance,
    student_t_expected_shortfall,
    student_t_quantile,
)
from .stress_testing import (
    DEFAULT_HISTORICAL_SCENARIOS,
    HistoricalStressResult,
    HistoricalStressScenario,
    HypotheticalShock,
    HypotheticalStressResult,
    StressTester,
    stressed_covariance,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "BacktestResult",
    "ChristoffersenResult",
    "KupiecResult",
    "backtest_summary_table",
    "christoffersen_test",
    "kupiec_pof_test",
    "rolling_var_backtest",
    "run_standard_backtests",
    "DEFAULT_START",
    "DEFAULT_TICKERS",
    "DataLoadError",
    "DataQualityReport",
    "MarketData",
    "generate_synthetic_market_data",
    "load_csv_prices",
    "load_market_data",
    "DEFAULT_WEIGHTS",
    "EULER_TOLERANCE",
    "EulerRiskDecomposer",
    "RiskDecomposition",
    "align_weights",
    "analytic_normal_euler",
    "empirical_component_cvar",
    "MODEL_KEYS",
    "MODEL_LABELS",
    "FilteredEWMAStudentTRiskModel",
    "HistoricalSimulationRiskModel",
    "InsufficientDataError",
    "MonteCarloNormalRiskModel",
    "MonteCarloStudentTRiskModel",
    "ParametricNormalRiskModel",
    "RiskMeasureResult",
    "RiskModel",
    "RiskModelError",
    "build_risk_models",
    "estimate_degrees_of_freedom",
    "evaluate_models",
    "ewma_covariance",
    "ewma_volatility",
    "sample_covariance",
    "student_t_expected_shortfall",
    "student_t_quantile",
    "DEFAULT_HISTORICAL_SCENARIOS",
    "HistoricalStressResult",
    "HistoricalStressScenario",
    "HypotheticalShock",
    "HypotheticalStressResult",
    "StressTester",
    "stressed_covariance",
]
