"""Market-data acquisition, cleaning and integrity validation.

The module supports three data sources, in order of preference:

1. A local CSV file (Bloomberg ``PX_LAST``-style wide export: ``Date`` + one
   column per asset).
2. Yahoo Finance via :mod:`yfinance` for the default multi-asset portfolio.
3. A deterministic synthetic panel used when no network access is available.
   The synthetic panel embeds the 2008, 2020 and 2022 crisis regimes so that
   stress replays remain meaningful in offline mode.

All cleaning is strictly causal: prices are only ever forward-filled from the
past, never back-filled, so return series contain no look-ahead bias.

Sign conventions
----------------
Prices are strictly positive. Simple returns are used for P&L arithmetic
(``w @ r`` is exact for a portfolio marked to market at ``t``); log returns are
provided for distributional analysis and volatility work.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "TLT", "IEF", "EURUSD=X", "GLD")
DEFAULT_START: str = "2004-01-01"
MIN_OBSERVATIONS: int = 60
MAX_FORWARD_FILL: int = 5
MISSING_RATIO_LIMIT: float = 0.05
EXTREME_RETURN_LIMIT: float = 0.25
STALE_RUN_LIMIT: int = 10

_STALE_SUFFIXES: tuple[str, ...] = (
    " US EQUITY",
    " EQUITY",
    " CURNCY",
    " CURRENCY",
    " COMDTY",
    " INDEX",
    " US",
)

_NULL_TOKENS: tuple[str, ...] = (
    "",
    "-",
    "--",
    "N/A",
    "N.A.",
    "#N/A",
    "#N/A N/A",
    "NAN",
    "NULL",
    "NONE",
)


class DataLoadError(RuntimeError):
    """Raised when market data cannot be obtained or fails validation."""


@dataclass(frozen=True)
class DataQualityReport:
    """Immutable snapshot of the integrity checks applied to a price panel.

    Attributes:
        source: Human-readable description of where the data came from.
        tickers: Asset identifiers kept after cleaning.
        start: First observation timestamp.
        end: Last observation timestamp.
        observations: Number of rows in the cleaned panel.
        missing_ratio: Share of cells that were missing or non-positive
            before cleaning.
        non_positive_cells: Cells discarded because price <= 0.
        dropped_rows: Rows removed because at least one asset had no price.
        stale_assets: Assets with long runs of unchanged prices.
        extreme_assets: Assets with an absolute daily return above
            :data:`EXTREME_RETURN_LIMIT`.
        warnings: Non-fatal integrity warnings.
        passed: ``True`` when the panel clears the minimum quality bar.
    """

    source: str
    tickers: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    observations: int
    missing_ratio: float
    non_positive_cells: int
    dropped_rows: int
    stale_assets: tuple[str, ...]
    extreme_assets: tuple[str, ...]
    warnings: tuple[str, ...]
    passed: bool

    def to_frame(self) -> pd.DataFrame:
        """Return a one-row tabular summary suitable for reporting."""
        return pd.DataFrame(
            [
                {
                    "source": self.source,
                    "assets": ", ".join(self.tickers),
                    "start": self.start.date().isoformat(),
                    "end": self.end.date().isoformat(),
                    "observations": self.observations,
                    "missing_ratio": round(self.missing_ratio, 6),
                    "non_positive_cells": self.non_positive_cells,
                    "dropped_rows": self.dropped_rows,
                    "stale_assets": ", ".join(self.stale_assets) or "-",
                    "extreme_assets": ", ".join(self.extreme_assets) or "-",
                    "passed": self.passed,
                }
            ]
        )

    def summary(self) -> str:
        """Return a compact single-line description of the panel."""
        return (
            f"{self.observations} observations "
            f"({self.start.date()} to {self.end.date()}) from {self.source}; "
            f"missing={self.missing_ratio:.2%}; passed={self.passed}"
        )


@dataclass(frozen=True)
class MarketData:
    """Cleaned price panel plus derived return matrices.

    Attributes:
        prices: Daily closing prices, strictly positive, one column per asset.
        source: Description of the data provenance.
        quality: Integrity report produced during cleaning.
    """

    prices: pd.DataFrame
    source: str
    quality: DataQualityReport

    def __post_init__(self) -> None:
        if self.prices.empty:
            raise DataLoadError("Price panel is empty.")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise DataLoadError("Price panel must be indexed by dates.")
        if self.prices.index.has_duplicates:
            raise DataLoadError("Price panel index contains duplicate dates.")
        if not self.prices.index.is_monotonic_increasing:
            raise DataLoadError("Price panel index must be sorted ascending.")
        values = self.prices.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataLoadError("Price panel contains non-finite values.")
        if (values <= 0).any():
            raise DataLoadError("Price panel contains non-positive prices.")
        if self.prices.shape[1] < 1:
            raise DataLoadError("Price panel must contain at least one asset.")

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    @property
    def tickers(self) -> tuple[str, ...]:
        """Asset identifiers in column order."""
        return tuple(str(c) for c in self.prices.columns)

    @property
    def n_observations(self) -> int:
        """Number of price observations."""
        return int(len(self.prices))

    @property
    def start(self) -> pd.Timestamp:
        """First price date."""
        return pd.Timestamp(self.prices.index[0])

    @property
    def end(self) -> pd.Timestamp:
        """Last price date."""
        return pd.Timestamp(self.prices.index[-1])

    @property
    def latest_prices(self) -> pd.Series:
        """Most recent closing price per asset."""
        return self.prices.iloc[-1].astype(float)

    @property
    def simple_returns(self) -> pd.DataFrame:
        """Arithmetic daily returns with the first row dropped."""
        return self.prices.pct_change(fill_method=None).dropna(how="any")

    @property
    def log_returns(self) -> pd.DataFrame:
        """Continuously compounded daily returns with the first row dropped."""
        return np.log(self.prices).diff().dropna(how="any")

    def slice(self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None) -> "MarketData":
        """Return a causal sub-panel over ``[start, end]`` (inclusive).

        One price observation before ``start`` is retained so that the first
        return inside the window is computed from information known at the
        window's first trading day. This avoids look-ahead bias.

        Args:
            start: First date of the window, or ``None`` for the panel start.
            end: Last date of the window, or ``None`` for the panel end.

        Returns:
            A new :class:`MarketData` restricted to the window.

        Raises:
            DataLoadError: If the window contains fewer than two prices.
        """
        index = self.prices.index
        lo = 0 if start is None else max(0, int(index.searchsorted(pd.Timestamp(start), side="left")) - 1)
        hi = len(index) if end is None else int(index.searchsorted(pd.Timestamp(end), side="right"))
        if hi - lo < 2:
            raise DataLoadError(f"Window {start}..{end} contains fewer than two price observations.")
        subset = self.prices.iloc[lo:hi].copy()
        label = f"{self.source} [{subset.index[0].date()}..{subset.index[-1].date()}]"
        return MarketData(prices=subset, source=label, quality=_quality_report(subset, label))

    def subset(self, tickers: Sequence[str]) -> "MarketData":
        """Return a panel restricted to ``tickers`` while preserving order.

        Args:
            tickers: Asset identifiers to keep.

        Raises:
            DataLoadError: If a requested ticker is absent.
        """
        missing = [t for t in tickers if t not in self.prices.columns]
        if missing:
            raise DataLoadError(f"Requested tickers missing from panel: {missing}")
        subset = self.prices.loc[:, list(tickers)].copy()
        label = f"{self.source} [subset]"
        return MarketData(prices=subset, source=label, quality=_quality_report(subset, label))


# ----------------------------------------------------------------------
# Cleaning helpers
# ----------------------------------------------------------------------
def normalise_ticker(name: object) -> str:
    """Normalise a raw column label into a plain ticker symbol.

    Handles Bloomberg suffixes such as ``SPY US Equity`` or ``EURUSD CURNCY``
    as well as plain Yahoo Finance symbols (``EURUSD=X``).

    Args:
        name: Raw column label.

    Returns:
        Upper-case symbol without venue/asset-class suffixes.
    """
    token = " ".join(str(name).strip().split()).upper()
    for suffix in _STALE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            token = token[: -len(suffix)].strip()
    return token


def _to_datetime_index(values: pd.Series) -> pd.DatetimeIndex:
    """Parse a date-like column into a ``DatetimeIndex``.

    Tries ISO/american formats first, then day-first, then ``YYYYMMDD``
    integers, which appear in older Bloomberg exports.
    """
    parsed = pd.to_datetime(values.astype(str), errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(values.astype(str), errors="coerce", dayfirst=True)
    if parsed.isna().all():
        parsed = pd.to_datetime(values.astype(str), errors="coerce", format="%Y%m%d")
    return pd.DatetimeIndex(parsed)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every cell to float, mapping textual nulls to ``NaN``."""
    cleaned = frame.astype(str).apply(lambda col: col.str.strip())
    cleaned = cleaned.replace({token: np.nan for token in _NULL_TOKENS})
    cleaned = cleaned.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan})
    cleaned = cleaned.apply(lambda col: col.str.replace(",", "", regex=False))
    return cleaned.apply(pd.to_numeric, errors="coerce")


def _stale_assets(prices: pd.DataFrame, min_run: int = STALE_RUN_LIMIT) -> tuple[str, ...]:
    """Identify assets whose price is unchanged for ``min_run`` consecutive days."""
    flagged: list[str] = []
    for column in prices.columns:
        values = prices[column].to_numpy(dtype=float)
        longest = 1
        run = 1
        for i in range(1, len(values)):
            if values[i] == values[i - 1]:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        if longest >= min_run:
            flagged.append(str(column))
    return tuple(flagged)


def _extreme_assets(prices: pd.DataFrame, limit: float = EXTREME_RETURN_LIMIT) -> tuple[str, ...]:
    """Identify assets with an absolute daily return above ``limit``."""
    returns = prices.pct_change(fill_method=None)
    if returns.empty:
        return ()
    peaks = returns.abs().max(axis=0, skipna=True)
    return tuple(str(c) for c in peaks.index[peaks > limit])


def _quality_report(
    prices: pd.DataFrame,
    source: str,
    *,
    missing_ratio: float = 0.0,
    non_positive_cells: int = 0,
    dropped_rows: int = 0,
    warnings: tuple[str, ...] = (),
) -> DataQualityReport:
    """Build a :class:`DataQualityReport` for an already-cleaned panel."""
    observation_count = int(len(prices))
    stale = _stale_assets(prices)
    extreme = _extreme_assets(prices)
    notes = list(warnings)
    if stale:
        notes.append(f"Stale prices detected for {', '.join(stale)}.")
    if extreme:
        notes.append(f"Daily moves above {EXTREME_RETURN_LIMIT:.0%} for {', '.join(extreme)}.")
    passed = (
        observation_count >= MIN_OBSERVATIONS
        and prices.shape[1] >= 1
        and missing_ratio <= MISSING_RATIO_LIMIT
        and bool(np.isfinite(prices.to_numpy(dtype=float)).all())
    )
    return DataQualityReport(
        source=source,
        tickers=tuple(str(c) for c in prices.columns),
        start=pd.Timestamp(prices.index[0]),
        end=pd.Timestamp(prices.index[-1]),
        observations=observation_count,
        missing_ratio=float(missing_ratio),
        non_positive_cells=int(non_positive_cells),
        dropped_rows=int(dropped_rows),
        stale_assets=stale,
        extreme_assets=extreme,
        warnings=tuple(notes),
        passed=bool(passed),
    )


def clean_price_panel(
    raw: pd.DataFrame,
    requested: Sequence[str] | None = None,
    *,
    source: str = "unknown",
    min_observations: int = MIN_OBSERVATIONS,
    max_forward_fill: int = MAX_FORWARD_FILL,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Clean and validate a raw wide price panel.

    Steps:
        1. Normalise column labels and drop duplicate assets.
        2. Coerce values to float, mapping textual nulls to ``NaN``.
        3. Treat non-positive prices as missing.
        4. Forward-fill at most ``max_forward_fill`` observations (causal).
        5. Drop rows that still contain a missing price.
        6. Validate positivity, monotonicity and minimum history.

    Args:
        raw: Raw wide DataFrame whose index is date-like.
        requested: Optional list of tickers expected in the output.
        source: Provenance label for the report.
        min_observations: Minimum acceptable number of rows.
        max_forward_fill: Maximum consecutive forward-filled observations.

    Returns:
        A tuple ``(prices, report)``.

    Raises:
        DataLoadError: If the panel is unusable or requested assets are absent.
    """
    frame = raw.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [normalise_ticker(column[-1]) for column in frame.columns]
    else:
        frame.columns = [normalise_ticker(column) for column in frame.columns]

    frame = frame.loc[:, [c for c in frame.columns if c not in {"", "DATE", "DATES"}]]
    if frame.columns.has_duplicates:
        frame = frame.T.groupby(level=0).last().T
    if frame.shape[1] == 0:
        raise DataLoadError("No asset columns found in price panel.")

    frame = _numeric_frame(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame[~pd.isna(frame.index)]
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(how="all")

    total_cells = int(frame.size)
    missing_cells = int(frame.isna().sum().sum())
    non_positive = int((frame <= 0).sum().sum())
    frame = frame.mask(frame <= 0)

    dropped_before = int(len(frame))
    frame = frame.ffill(limit=max_forward_fill)
    frame = frame.dropna(how="any")
    dropped_rows = dropped_before - int(len(frame))
    missing_ratio = 0.0 if total_cells == 0 else (missing_cells + non_positive) / total_cells

    if requested is not None:
        wanted = [normalise_ticker(t) for t in requested]
        available = [str(column) for column in frame.columns]
        resolved: list[str] = []
        absent: list[str] = []
        for ticker in wanted:
            if ticker in available:
                resolved.append(ticker)
            elif ticker.endswith("=X") and ticker[:-2] in available:
                # Bloomberg exports FX as ``EURUSD CURNCY`` -> ``EURUSD`` while
                # the default universe uses Yahoo's ``EURUSD=X`` spelling.
                resolved.append(ticker[:-2])
            elif f"{ticker}=X" in available:
                resolved.append(f"{ticker}=X")
            else:
                absent.append(ticker)
        if absent:
            raise DataLoadError(f"Price panel is missing requested assets: {absent}")
        if len(set(resolved)) != len(resolved):
            raise DataLoadError("Requested tickers resolve to duplicate columns in the price panel.")
        frame = frame.loc[:, resolved].copy()
        # Present assets under the caller's canonical identifiers.
        frame.columns = wanted

    if len(frame) < min_observations:
        raise DataLoadError(
            f"Insufficient history: {len(frame)} observations after cleaning, "
            f"minimum is {min_observations}."
        )

    # Drop assets that carry no information (constant price).
    variances = frame.pct_change(fill_method=None).var(axis=0, skipna=True)
    zero_variance = [str(c) for c in variances.index if not np.isfinite(variances[c]) or variances[c] <= 0]
    if zero_variance:
        LOGGER.warning("Dropping zero-variance assets: %s", zero_variance)
        frame = frame.drop(columns=zero_variance)
        if frame.shape[1] == 0:
            raise DataLoadError("All assets were removed as zero-variance series.")

    report = _quality_report(
        frame,
        source,
        missing_ratio=missing_ratio,
        non_positive_cells=non_positive,
        dropped_rows=dropped_rows,
    )
    if not report.passed:
        raise DataLoadError(f"Data quality gate failed: {report.summary()}")
    return frame, report


# ----------------------------------------------------------------------
# CSV (Bloomberg-style) loading
# ----------------------------------------------------------------------
def load_csv_prices(
    path: str | Path,
    tickers: Sequence[str] | None = None,
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load a wide price panel from a local CSV file.

    The file must contain a date column first, followed by one column per
    asset. Bloomberg exports such as ``SPY US Equity`` and Yahoo symbols are
    both accepted. Textual nulls (``#N/A N/A``, ``N/A``, ``-``) are treated as
    missing. A stray field-name row (``PX_LAST``) is discarded automatically.

    Args:
        path: CSV location.
        tickers: Optional asset list to retain and order.
        min_observations: Minimum number of observations required.

    Returns:
        Tuple ``(prices, report)``.

    Raises:
        DataLoadError: If the file is missing, malformed or fails validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataLoadError(f"CSV file not found: {file_path}")
    try:
        frame = pd.read_csv(file_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise DataLoadError(f"Could not parse CSV {file_path}: {exc}") from exc
    if frame.empty or frame.shape[1] < 2:
        raise DataLoadError(f"CSV {file_path} must contain a date column and at least one asset column.")

    date_column = frame.columns[0]
    index = _to_datetime_index(frame[date_column])
    frame = frame.iloc[:, 1:]
    frame.index = index
    frame = frame.loc[~pd.isna(frame.index)]

    source = f"CSV {file_path.name}"
    prices, report = clean_price_panel(
        frame,
        tickers,
        source=source,
        min_observations=min_observations,
    )
    LOGGER.info("Loaded %s", report.summary())
    return prices, report


# ----------------------------------------------------------------------
# Yahoo Finance loading
# ----------------------------------------------------------------------
def _download_yfinance(
    tickers: Sequence[str],
    start: str,
    end: str | None,
    *,
    retries: int = 3,
    pause: float = 2.0,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance with retries."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataLoadError("yfinance is not installed.") from exc

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                list(tickers),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if raw is None or raw.empty:
                raise DataLoadError("Yahoo Finance returned an empty frame.")
            if isinstance(raw.columns, pd.MultiIndex):
                levels = raw.columns.get_level_values(0)
                if "Close" in levels:
                    close = raw["Close"]
                else:
                    close = raw.xs(levels[0], axis=1, level=0)
            else:
                close = raw
            if isinstance(close, pd.Series):
                close = close.to_frame(name=normalise_ticker(tickers[0]))
            close = close.apply(pd.to_numeric, errors="coerce")
            close = close.dropna(how="all")
            if close.empty:
                raise DataLoadError("Yahoo Finance returned no usable close prices.")
            return close
        except Exception as exc:  # network, rate limit, provider hiccup
            last_error = exc
            LOGGER.warning(
                "Yahoo Finance download attempt %d/%d failed: %s", attempt, retries, exc
            )
            if attempt < retries:
                time.sleep(pause * attempt)
    raise DataLoadError(f"Yahoo Finance download failed after {retries} attempts: {last_error}")


# ----------------------------------------------------------------------
# Synthetic offline panel
# ----------------------------------------------------------------------
_SYNTHETIC_VOL: dict[str, float] = {
    "SPY": 0.19,
    "QQQ": 0.26,
    "TLT": 0.14,
    "IEF": 0.07,
    "EURUSD=X": 0.085,
    "GLD": 0.16,
}
_SYNTHETIC_DRIFT: dict[str, float] = {
    "SPY": 0.09,
    "QQQ": 0.12,
    "TLT": 0.035,
    "IEF": 0.02,
    "EURUSD=X": 0.0,
    "GLD": 0.055,
}
_SYNTHETIC_PRICE0: dict[str, float] = {
    "SPY": 120.0,
    "QQQ": 40.0,
    "TLT": 90.0,
    "IEF": 85.0,
    "EURUSD=X": 1.25,
    "GLD": 45.0,
}
# Pairwise correlations for the default portfolio (order follows the keys).
_SYNTHETIC_CORR_NAMES: tuple[str, ...] = ("SPY", "QQQ", "TLT", "IEF", "EURUSD=X", "GLD")
_SYNTHETIC_CORR = np.array(
    [
        [1.00, 0.92, -0.28, -0.22, 0.02, 0.05],
        [0.92, 1.00, -0.30, -0.24, 0.05, 0.03],
        [-0.28, -0.30, 1.00, 0.93, -0.15, 0.18],
        [-0.22, -0.24, 0.93, 1.00, -0.12, 0.16],
        [0.02, 0.05, -0.15, -0.12, 1.00, -0.35],
        [0.05, 0.03, 0.18, 0.16, -0.35, 1.00],
    ],
    dtype=float,
)

_SYNTHETIC_REGIMES: tuple[dict[str, object], ...] = (
    {
        "label": "2008 Global Financial Crisis",
        "start": "2008-09-01",
        "end": "2008-11-30",
        "vol_multiplier": 2.6,
        "targets": {
            "SPY": -0.32,
            "QQQ": -0.38,
            "TLT": 0.09,
            "IEF": 0.06,
            "EURUSD=X": -0.04,
            "GLD": -0.06,
        },
    },
    {
        "label": "2020 COVID-19 Liquidity Shock",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "vol_multiplier": 3.4,
        "targets": {
            "SPY": -0.34,
            "QQQ": -0.28,
            "TLT": 0.08,
            "IEF": 0.05,
            "EURUSD=X": -0.03,
            "GLD": -0.01,
        },
    },
    {
        "label": "2022 Inflation & Rate Shock",
        "start": "2022-01-03",
        "end": "2022-10-31",
        "vol_multiplier": 1.4,
        "targets": {
            "SPY": -0.19,
            "QQQ": -0.29,
            "TLT": -0.32,
            "IEF": -0.15,
            "EURUSD=X": -0.13,
            "GLD": -0.09,
        },
    },
)


def _synthetic_covariance(tickers: Sequence[str]) -> np.ndarray:
    """Build an annualised covariance matrix for arbitrary ticker lists."""
    n = len(tickers)
    vols = np.array([_SYNTHETIC_VOL.get(t, 0.15) for t in tickers], dtype=float)
    corr = np.full((n, n), 0.35, dtype=float)
    np.fill_diagonal(corr, 1.0)
    for i, first in enumerate(tickers):
        for j, second in enumerate(tickers):
            if i == j:
                continue
            if first in _SYNTHETIC_CORR_NAMES and second in _SYNTHETIC_CORR_NAMES:
                row = _SYNTHETIC_CORR_NAMES.index(first)
                col = _SYNTHETIC_CORR_NAMES.index(second)
                corr[i, j] = _SYNTHETIC_CORR[row, col]
    covariance = corr * np.outer(vols, vols)
    return 0.5 * (covariance + covariance.T)


def generate_synthetic_market_data(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    seed: int = 7,
) -> MarketData:
    """Generate a deterministic multi-asset panel for offline operation.

    The generator uses a three-factor correlation structure, Student-t shocks
    (5 degrees of freedom) for realistic tail dependence, and deterministic
    regime overlays that reproduce the cumulative moves of the 2008 GFC, the
    2020 COVID crash and the 2022 rate shock. Prices are a pure function of
    ``seed``, ``start`` and ``end``.

    Args:
        tickers: Asset symbols to simulate.
        start: First business day.
        end: Last business day (defaults to today).
        seed: Random seed for reproducibility.

    Returns:
        A validated :class:`MarketData` instance.

    Raises:
        DataLoadError: If fewer than two tickers are requested or the range is
            too short.
    """
    symbols = tuple(normalise_ticker(t) for t in tickers)
    if len(symbols) < 2:
        raise DataLoadError("Synthetic panels require at least two assets.")
    end_date = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    index = pd.bdate_range(start=start, end=end_date)
    if len(index) < MIN_OBSERVATIONS:
        raise DataLoadError("Synthetic date range is too short to be useful.")

    n_obs = len(index)
    n_assets = len(symbols)
    covariance = _synthetic_covariance(symbols)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() <= 0:  # pragma: no cover - defensive
        covariance += np.eye(n_assets) * (abs(eigenvalues.min()) + 1e-8)
    cholesky = np.linalg.cholesky(covariance)

    rng = np.random.default_rng(seed)
    normal = rng.standard_normal((n_obs, n_assets))
    chi_square = rng.chisquare(df=5, size=n_obs)
    t_scale = np.sqrt((5 - 2) / chi_square)  # unit-variance Student-t shocks
    shocks = (normal @ cholesky.T) * t_scale[:, None] / np.sqrt(252.0)

    drift = np.array([_SYNTHETIC_DRIFT.get(t, 0.04) for t in symbols], dtype=float) / 252.0
    log_returns = drift[None, :] + shocks

    for regime in _SYNTHETIC_REGIMES:
        regime_start = max(pd.Timestamp(str(regime["start"])), index[0])
        regime_end = min(pd.Timestamp(str(regime["end"])), index[-1])
        if regime_start >= regime_end:
            continue
        mask = (index >= regime_start) & (index <= regime_end)
        count = int(mask.sum())
        if count < 2:
            continue
        multiplier = float(regime["vol_multiplier"])
        log_returns[mask, :] *= multiplier
        targets = regime["targets"]
        assert isinstance(targets, dict)
        for position, symbol in enumerate(symbols):
            target = targets.get(symbol)
            if target is None:
                continue
            current = float(log_returns[mask, position].sum())
            desired = float(np.log1p(target))
            log_returns[mask, position] += (desired - current) / count

    level0 = np.array([_SYNTHETIC_PRICE0.get(t, 100.0) for t in symbols], dtype=float)
    prices = pd.DataFrame(
        level0[None, :] * np.exp(np.cumsum(log_returns, axis=0)),
        index=index,
        columns=list(symbols),
    )
    source = "synthetic offline panel (seeded regime model)"
    cleaned, report = clean_price_panel(prices, symbols, source=source)
    LOGGER.info("Generated %s", report.summary())
    return MarketData(prices=cleaned, source=source, quality=report)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def load_market_data(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    *,
    csv_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    offline: bool = False,
    allow_synthetic: bool = True,
    synthetic_seed: int = 7,
    min_observations: int = MIN_OBSERVATIONS,
) -> MarketData:
    """Load a cleaned multi-asset price panel.

    Resolution order: an explicit ``csv_path`` (authoritative, no fallback),
    then Yahoo Finance, then the synthetic offline panel when
    ``allow_synthetic`` is true.

    Args:
        tickers: Asset symbols to load.
        csv_path: Optional local CSV panel.
        start: History start date (defaults to :data:`DEFAULT_START`).
        end: History end date (defaults to today).
        offline: Skip network access and use the synthetic panel.
        allow_synthetic: Fall back to synthetic data if Yahoo Finance fails.
        synthetic_seed: Seed for the synthetic panel.
        min_observations: Minimum number of observations required.

    Returns:
        A :class:`MarketData` instance.

    Raises:
        DataLoadError: If no usable data source is available.
    """
    symbols = tuple(normalise_ticker(t) for t in tickers)
    if len(symbols) < 2:
        raise DataLoadError("At least two assets are required for a portfolio risk engine.")
    start_date = start or DEFAULT_START

    if csv_path is not None:
        prices, report = load_csv_prices(csv_path, symbols, min_observations=min_observations)
        return MarketData(prices=prices, source=str(report.source), quality=report)

    if offline:
        return generate_synthetic_market_data(symbols, start=start_date, end=end, seed=synthetic_seed)

    try:
        raw = _download_yfinance(symbols, start_date, end)
        available = [t for t in symbols if t in {normalise_ticker(c) for c in raw.columns}]
        absent = [t for t in symbols if t not in available]
        if absent:
            LOGGER.warning("Yahoo Finance did not return prices for %s; dropping them.", absent)
        if len(available) < 2:
            raise DataLoadError(f"Yahoo Finance returned fewer than two requested assets (missing {absent}).")
        prices, report = clean_price_panel(
            raw,
            available,
            source="Yahoo Finance (adjusted close)",
            min_observations=min_observations,
        )
        return MarketData(prices=prices, source=str(report.source), quality=report)
    except DataLoadError as exc:
        if not allow_synthetic:
            raise
        LOGGER.warning("Falling back to synthetic data: %s", exc)
        return generate_synthetic_market_data(symbols, start=start_date, end=end, seed=synthetic_seed)


def fetch_prices(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper returning only the price frame.

    Args:
        tickers: Asset symbols.
        start: History start date.
        end: History end date.

    Returns:
        Cleaned daily close prices.
    """
    return load_market_data(tickers, start=start, end=end).prices
