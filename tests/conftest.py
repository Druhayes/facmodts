"""
Pytest configuration and fixtures for facmodts testing.

This module provides:
- Session-scoped fixtures for test data
- Hierarchical tolerance specifications for R comparison
- Synthetic data generators
- R integration fixtures (rpy2)
"""

from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
import pytest

# Try to import rpy2 for R comparison tests
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


@pytest.fixture(scope="session")
def tolerances() -> Dict[str, Dict[str, float]]:
    """
    Hierarchical tolerances for R-Python comparison tests.

    Different operations have different numerical precision requirements.
    Beta coefficients should match very closely, while risk measures
    (VaR, ES) may have larger differences due to kernel estimation.

    Returns:
        Dictionary mapping operation type to relative/absolute tolerances.
    """
    return {
        "beta": {"rtol": 1e-8, "atol": 1e-10},
        "alpha": {"rtol": 1e-8, "atol": 1e-10},
        "r_squared": {"rtol": 1e-6, "atol": 1e-8},
        "residuals": {"rtol": 1e-7, "atol": 1e-9},
        "resid_sd": {"rtol": 1e-6, "atol": 1e-8},
        "sd_decomp": {"rtol": 1e-4, "atol": 1e-6},
        "var_decomp": {"rtol": 1e-3, "atol": 1e-5},
        "es_decomp": {"rtol": 1e-3, "atol": 1e-5},
        "covariance": {"rtol": 1e-5, "atol": 1e-7},
    }


@pytest.fixture(scope="session")
def stocks_factors_csv() -> pl.DataFrame:
    """
    Load the full stocks_factors.csv dataset (294 assets x 60 periods).

    This real-world dataset is used for R comparison tests to avoid
    random seed differences between R and Python.

    Returns:
        Polars DataFrame with stock returns and factor exposures.
    """
    csv_path = Path(__file__).parent / "data" / "stocks_factors.csv"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")

    return pl.read_csv(csv_path)


@pytest.fixture(scope="session")
def stocks_factors_subset(stocks_factors_csv: pl.DataFrame) -> pl.DataFrame:
    """
    50-asset subset of stocks_factors.csv for fast unit tests.

    Using a subset speeds up tests during development while still
    providing real data for validation.

    Returns:
        Polars DataFrame with 50 assets x 60 periods.
    """
    unique_tickers = stocks_factors_csv["TickerLast"].unique()[:50]
    return stocks_factors_csv.filter(pl.col("TickerLast").is_in(unique_tickers))


@pytest.fixture(scope="session")
def r_facmodts():
    """
    Load the R facmodTS package for comparison testing.

    Requires rpy2 and R with facmodTS installed.

    Returns:
        rpy2 R interface with facmodTS loaded.

    Raises:
        pytest.skip: If rpy2 is not available or R package cannot be loaded.
    """
    if not HAS_RPY2:
        pytest.skip("rpy2 not available for R comparison tests")

    try:
        # Activate pandas conversion
        pandas2ri.activate()

        # Load R facmodTS package
        ro.r('library(facmodTS)')

        return ro.r
    except Exception as e:
        pytest.skip(f"Could not load R facmodTS package: {e}")


@pytest.fixture
def synthetic_returns_simple() -> pl.DataFrame:
    """
    Generate simple synthetic returns for basic unit tests.

    Returns 3 assets, 3 factors, 100 time periods with known relationships.
    No random seed needed - deterministic data.

    Returns:
        Polars DataFrame with columns: date, Asset1, Asset2, Asset3,
        Factor1, Factor2, Factor3.
    """
    n_periods = 100

    # Generate date index
    from datetime import date, timedelta
    start_date = date(2020, 1, 1)
    end_date = start_date + timedelta(days=n_periods - 1)
    dates = pl.date_range(start_date, end_date, interval="1d", eager=True)

    # Generate factors (deterministic)
    factor1 = 0.01 + 0.02 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
    factor2 = -0.005 + 0.015 * np.cos(np.linspace(0, 6 * np.pi, n_periods))
    factor3 = 0.003 * np.linspace(-1, 1, n_periods)

    # Generate assets with known betas
    # Asset1: beta = [0.8, 0.3, 0.0], alpha = 0.001
    # Asset2: beta = [1.2, -0.5, 0.4], alpha = 0.002
    # Asset3: beta = [0.5, 0.0, 0.8], alpha = -0.001

    asset1 = 0.001 + 0.8 * factor1 + 0.3 * factor2 + 0.0 * factor3 + 0.005 * np.random.randn(n_periods)
    asset2 = 0.002 + 1.2 * factor1 - 0.5 * factor2 + 0.4 * factor3 + 0.007 * np.random.randn(n_periods)
    asset3 = -0.001 + 0.5 * factor1 + 0.0 * factor2 + 0.8 * factor3 + 0.004 * np.random.randn(n_periods)

    df = pl.DataFrame(
        {
            "date": dates,
            "Asset1": asset1,
            "Asset2": asset2,
            "Asset3": asset3,
            "Factor1": factor1,
            "Factor2": factor2,
            "Factor3": factor3,
        }
    )

    return df


@pytest.fixture
def synthetic_returns_with_rf() -> pl.DataFrame:
    """
    Generate synthetic returns including risk-free rate.

    Returns:
        Polars DataFrame with asset returns, factor returns, and RF column.
    """
    n_periods = 100

    # Generate date index
    from datetime import date, timedelta
    start_date = date(2020, 1, 1)
    end_date = start_date + timedelta(days=n_periods - 1)
    dates = pl.date_range(start_date, end_date, interval="1d", eager=True)

    # Risk-free rate (constant)
    rf = np.full(n_periods, 0.0003)  # ~0.03% per period

    # Generate factors
    factor1 = 0.01 + 0.02 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
    factor2 = -0.005 + 0.015 * np.cos(np.linspace(0, 6 * np.pi, n_periods))

    # Generate assets (total returns, not excess)
    # asset = alpha + beta * factor + noise (total returns)
    np.random.seed(42)  # For reproducibility
    asset1 = 0.0005 + 0.8 * factor1 + 0.005 * np.random.randn(n_periods)
    asset2 = 0.001 + 1.2 * factor1 - 0.3 * factor2 + 0.007 * np.random.randn(n_periods)

    df = pl.DataFrame(
        {
            "date": dates,
            "Asset1": asset1,
            "Asset2": asset2,
            "Factor1": factor1,
            "Factor2": factor2,
            "RF": rf,
        }
    )

    return df


@pytest.fixture
def synthetic_returns_with_nas() -> pl.DataFrame:
    """
    Generate synthetic returns with some NA values to test incomplete case handling.

    Returns:
        Polars DataFrame with missing values in various locations.
    """
    n_periods = 100

    from datetime import date, timedelta
    start_date = date(2020, 1, 1)
    end_date = start_date + timedelta(days=n_periods - 1)
    dates = pl.date_range(start_date, end_date, interval="1d", eager=True)

    # Generate complete data first
    factor1 = 0.01 + 0.02 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
    factor2 = -0.005 + 0.015 * np.cos(np.linspace(0, 6 * np.pi, n_periods))

    asset1 = 0.001 + 0.8 * factor1 + 0.3 * factor2 + 0.005 * np.random.randn(n_periods)
    asset2 = 0.002 + 1.2 * factor1 - 0.5 * factor2 + 0.007 * np.random.randn(n_periods)

    # Create DataFrame first
    df = pl.DataFrame(
        {
            "date": dates,
            "Asset1": asset1,
            "Asset2": asset2,
            "Factor1": factor1,
            "Factor2": factor2,
        }
    )

    # Introduce NAs using Polars (None, not np.nan)
    df = df.with_columns([
        pl.when(pl.col("date").is_between(dates[5], dates[9], closed="both"))
        .then(None)
        .otherwise(pl.col("Asset1"))
        .alias("Asset1"),
        pl.when(pl.col("date").is_between(dates[15], dates[17], closed="both"))
        .then(None)
        .otherwise(pl.col("Factor1"))
        .alias("Factor1"),
        pl.when(pl.col("date").is_between(dates[25], dates[29], closed="both"))
        .then(None)
        .otherwise(pl.col("Asset2"))
        .alias("Asset2"),
    ])

    return df


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "r_comparison: marks tests that require R (rpy2)")
    config.addinivalue_line("markers", "slow: marks tests as slow (> 1 second)")
