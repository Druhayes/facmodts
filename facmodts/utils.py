"""
Utility functions for facmodts package.

This module provides helper functions for data validation, preprocessing,
and numerical computations. Many performance-critical functions are
JIT-compiled using Numba.
"""

from typing import List, Optional, Tuple, Union

import numba as nb
import numpy as np
import polars as pl


def validate_column_names(data: pl.DataFrame, columns: List[str], context: str = "") -> None:
    """
    Validate that specified columns exist in the DataFrame.

    Parameters:
        data: Polars DataFrame to check.
        columns: List of column names that must exist.
        context: Context string for error message (e.g., "asset.names").

    Raises:
        ValueError: If any column is missing from data.
    """
    missing = [col for col in columns if col not in data.columns]
    if missing:
        ctx = f" for {context}" if context else ""
        raise ValueError(f"Columns not found in data{ctx}: {missing}")


def make_syntactically_valid(names: List[str]) -> List[str]:
    """
    Convert names to syntactically valid identifiers (R-style).

    Replaces spaces with periods and ensures names are valid Python identifiers
    (similar to R's make.names behavior for xts compatibility).

    Parameters:
        names: List of column names.

    Returns:
        List of syntactically valid names.

    Examples:
        >>> make_syntactically_valid(["Asset 1", "Factor-A", "Market"])
        ['Asset.1', 'Factor.A', 'Market']
    """
    valid_names = []
    for name in names:
        # Replace spaces and hyphens with periods (R xts behavior)
        valid = name.replace(" ", ".").replace("-", ".")
        valid_names.append(valid)
    return valid_names


def compute_excess_returns(
    data: pl.DataFrame, asset_names: List[str], factor_names: List[str], rf_name: str
) -> pl.DataFrame:
    """
    Compute excess returns by subtracting risk-free rate.

    Parameters:
        data: Polars DataFrame with return columns.
        asset_names: List of asset return column names.
        factor_names: List of factor return column names.
        rf_name: Name of risk-free rate column.

    Returns:
        Polars DataFrame with excess returns (original columns replaced).
    """
    result = data.clone()

    # Subtract rf from all assets
    for asset in asset_names:
        result = result.with_columns(
            (pl.col(asset) - pl.col(rf_name)).alias(asset)
        )

    # Subtract rf from all factors
    for factor in factor_names:
        result = result.with_columns(
            (pl.col(factor) - pl.col(rf_name)).alias(factor)
        )

    return result


def remove_incomplete_cases(
    data: pl.DataFrame, asset_name: str, factor_names: List[str]
) -> pl.DataFrame:
    """
    Remove rows with missing values for a specific asset and its factors.

    This function mimics R's na.omit behavior, removing incomplete cases
    on a per-asset basis to allow assets with unequal histories.

    Parameters:
        data: Polars DataFrame with time series data.
        asset_name: Name of the asset column.
        factor_names: List of factor column names.

    Returns:
        Polars DataFrame with complete cases only.
    """
    # Columns to check for NA
    check_cols = [asset_name] + factor_names

    # Filter out rows with any null values in these columns
    result = data.filter(
        pl.all_horizontal([pl.col(col).is_not_null() for col in check_cols])
    )

    return result


@nb.jit(nopython=True, cache=True)
def compute_dls_weights(n_obs: int, decay: float) -> np.ndarray:
    """
    Compute exponentially decaying weights for discounted least squares.

    Weights are computed as: w_t = decay^(n-t) / sum(decay^(n-i))
    where t = 0, 1, ..., n-1 (oldest to newest).

    The weights sum to 1.0 and give more weight to recent observations.

    Parameters:
        n_obs: Number of observations.
        decay: Decay factor in (0, 1].

    Returns:
        Array of weights with shape (n_obs,), normalized to sum to 1.

    Examples:
        >>> weights = compute_dls_weights(5, 0.95)
        >>> weights  # doctest: +SKIP
        array([0.139, 0.146, 0.154, 0.162, 0.171])  # Increasing weights
    """
    weights = np.zeros(n_obs, dtype=np.float64)

    # Compute unnormalized weights: decay^(n-1-i) for i = 0, ..., n-1
    for i in range(n_obs):
        weights[i] = decay ** (n_obs - 1 - i)

    # Normalize to sum to 1
    weights = weights / weights.sum()

    return weights


@nb.jit(nopython=True, cache=True)
def huber_loss(residuals: np.ndarray, c: float = 1.345) -> np.ndarray:
    """
    Huber loss function for robust estimation.

    Loss is quadratic for small residuals and linear for large residuals:
    - rho(r) = r^2 / 2         if |r| <= c
    - rho(r) = c|r| - c^2/2    if |r| > c

    Parameters:
        residuals: Array of residuals.
        c: Tuning constant (default 1.345 for 95% efficiency).

    Returns:
        Array of Huber losses.
    """
    loss = np.zeros_like(residuals, dtype=np.float64)

    for i in range(len(residuals)):
        r = residuals[i]
        abs_r = abs(r)

        if abs_r <= c:
            loss[i] = 0.5 * r * r
        else:
            loss[i] = c * abs_r - 0.5 * c * c

    return loss


@nb.jit(nopython=True, cache=True)
def huber_psi(residuals: np.ndarray, c: float = 1.345) -> np.ndarray:
    """
    Huber psi function (derivative of rho) for IRWLS.

    psi(r) = r         if |r| <= c
    psi(r) = c*sign(r) if |r| > c

    Parameters:
        residuals: Array of residuals.
        c: Tuning constant.

    Returns:
        Array of psi values.
    """
    psi = np.zeros_like(residuals, dtype=np.float64)

    for i in range(len(residuals)):
        r = residuals[i]
        abs_r = abs(r)

        if abs_r <= c:
            psi[i] = r
        else:
            psi[i] = c * np.sign(r)

    return psi


@nb.jit(nopython=True, cache=True)
def bisquare_psi(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Tukey's bisquare psi function for robust estimation.

    psi(r) = r * (1 - (r/c)^2)^2  if |r| <= c
    psi(r) = 0                     if |r| > c

    Parameters:
        residuals: Array of residuals.
        c: Tuning constant (default 4.685 for 95% efficiency).

    Returns:
        Array of psi values.
    """
    psi = np.zeros_like(residuals, dtype=np.float64)

    for i in range(len(residuals)):
        r = residuals[i]
        abs_r = abs(r)

        if abs_r <= c:
            u = r / c
            psi[i] = r * (1 - u * u) ** 2
        else:
            psi[i] = 0.0

    return psi


def median_absolute_deviation(x: np.ndarray, constant: float = 1.4826) -> float:
    """
    Compute the Median Absolute Deviation (MAD).

    MAD = constant * median(|x - median(x)|)

    The default constant 1.4826 makes MAD a consistent estimator
    of standard deviation for normally distributed data.

    Parameters:
        x: Input array.
        constant: Scaling constant for consistency.

    Returns:
        MAD estimate of scale.
    """
    median = np.median(x)
    abs_dev = np.abs(x - median)
    mad = constant * np.median(abs_dev)
    return mad


def make_padded_dataframe(
    coef_dict: dict, index_name: str = "asset"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Create padded arrays from dictionary of coefficient vectors with unequal lengths.

    When variable selection produces different factors for different assets,
    this function creates a padded matrix where missing coefficients are NaN.

    Parameters:
        coef_dict: Dictionary mapping asset names to coefficient arrays/dicts.
        index_name: Name for the row index (default "asset").

    Returns:
        Tuple of (padded_matrix, row_names, column_names).
        - padded_matrix: 2D array with shape (n_assets, n_all_factors).
        - row_names: List of asset names.
        - column_names: List of all unique factor names.

    Examples:
        >>> coef_dict = {
        ...     "Asset1": {"alpha": 0.01, "Factor1": 0.5, "Factor2": 0.3},
        ...     "Asset2": {"alpha": 0.02, "Factor1": 0.7}  # Factor2 missing
        ... }
        >>> matrix, rows, cols = make_padded_dataframe(coef_dict)
        >>> cols
        ['alpha', 'Factor1', 'Factor2']
        >>> matrix[1, 2]  # Asset2, Factor2 -> NaN
        nan
    """
    # Get all unique factor names
    all_factors = set()
    for asset_coefs in coef_dict.values():
        if isinstance(asset_coefs, dict):
            all_factors.update(asset_coefs.keys())
        elif isinstance(asset_coefs, np.ndarray):
            # Assume ordered factors if array
            all_factors.update([f"Factor{i}" for i in range(len(asset_coefs))])

    all_factors = sorted(all_factors)  # Consistent ordering
    n_factors = len(all_factors)

    # Get asset names
    asset_names = list(coef_dict.keys())
    n_assets = len(asset_names)

    # Create padded matrix
    padded_matrix = np.full((n_assets, n_factors), np.nan, dtype=np.float64)

    for i, asset in enumerate(asset_names):
        coefs = coef_dict[asset]
        if isinstance(coefs, dict):
            for j, factor in enumerate(all_factors):
                if factor in coefs:
                    padded_matrix[i, j] = coefs[factor]
        elif isinstance(coefs, np.ndarray):
            # Assume factor order matches all_factors
            for j, val in enumerate(coefs):
                if j < n_factors:
                    padded_matrix[i, j] = val

    return padded_matrix, asset_names, all_factors


def convert_date_column(data: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """
    Convert date column to Date type if not already.

    Parameters:
        data: Polars DataFrame.
        date_col: Name of date column (default "date").

    Returns:
        DataFrame with date column converted to Date type.
    """
    if date_col in data.columns:
        if data[date_col].dtype != pl.Date:
            data = data.with_columns(pl.col(date_col).cast(pl.Date))
    return data


def check_positive_definite(matrix: np.ndarray, name: str = "matrix") -> bool:
    """
    Check if a matrix is positive definite using Cholesky decomposition.

    Parameters:
        matrix: Square matrix to check.
        name: Name of matrix for error messages.

    Returns:
        True if positive definite, False otherwise.

    Raises:
        ValueError: If matrix is not square.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")

    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
