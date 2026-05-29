"""
Robust regression utilities for facmodts package.

This module provides robust regression functionality using M-estimators.
Note: Python uses M-estimators (Huber, Bisquare) while R uses MM-estimators.
MM-estimators provide higher breakdown points but are not available in statsmodels.
"""

from typing import Literal, Optional

import numpy as np
import statsmodels.api as sm
from statsmodels.robust import norms


def get_robust_norm(
    family: Literal["bisquare", "huber", "hampel", "andrews"] = "bisquare",
    tuning_constant: Optional[float] = None
) -> norms.RobustNorm:
    """
    Get robust norm (loss function) for RLM.

    Parameters:
        family: Loss function family. Options:
                - "bisquare": Tukey's bisquare (default, matches R's bisquare)
                - "huber": Huber loss
                - "hampel": Hampel's function
                - "andrews": Andrews' wave
        tuning_constant: Tuning constant for the norm. If None, uses default
                        for the family:
                        - bisquare: 4.685 (95% efficiency)
                        - huber: 1.345 (95% efficiency)
                        - hampel: (2, 4, 8) default
                        - andrews: 1.339 (95% efficiency)

    Returns:
        RobustNorm object for use with statsmodels RLM.

    Examples:
        >>> # Bisquare with default tuning
        >>> norm = get_robust_norm("bisquare")
        >>>
        >>> # Huber with custom tuning
        >>> norm = get_robust_norm("huber", tuning_constant=1.5)
    """
    if family == "bisquare":
        # Tukey's bisquare (biweight)
        c = tuning_constant if tuning_constant is not None else 4.685
        return norms.TukeyBiweight(c=c)

    elif family == "huber":
        # Huber loss
        c = tuning_constant if tuning_constant is not None else 1.345
        return norms.HuberT(t=c)

    elif family == "hampel":
        # Hampel's function
        if tuning_constant is not None:
            # User provided single constant, use as middle point
            abc = (tuning_constant * 0.5, tuning_constant, tuning_constant * 2)
        else:
            abc = (2.0, 4.0, 8.0)  # Default
        return norms.Hampel(a=abc[0], b=abc[1], c=abc[2])

    elif family == "andrews":
        # Andrews' wave
        c = tuning_constant if tuning_constant is not None else 1.339
        return norms.AndrewWave(a=c)

    else:
        raise ValueError(
            f"Unknown robust family '{family}'. "
            "Options: 'bisquare', 'huber', 'hampel', 'andrews'"
        )


def fit_rlm(
    y: np.ndarray,
    X: np.ndarray,
    family: Literal["bisquare", "huber", "hampel", "andrews"] = "bisquare",
    tuning_constant: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-7
):
    """
    Fit robust linear model using M-estimators.

    This is a wrapper around statsmodels RLM that provides an interface
    similar to OLS/WLS.

    Parameters:
        y: Response variable (n_obs,)
        X: Design matrix with constant (n_obs, n_features)
        family: Robust loss function family
        tuning_constant: Tuning constant for loss function
        max_iter: Maximum IRWLS iterations
        tol: Convergence tolerance

    Returns:
        RLM regression results object (statsmodels RegressionResultsWrapper).

    Notes:
        - Uses IRWLS (Iteratively Reweighted Least Squares)
        - M-estimators (not MM-estimators like R's lmrobdetMM)
        - Breakdown point is lower than R's MM-estimators
        - For bisquare with default tuning: ~95% efficiency at normal distribution

    Examples:
        >>> X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
        >>> y = X @ [1, 2, -1] + 0.1 * np.random.randn(100)
        >>> result = fit_rlm(y, X)
        >>> print(result.params)
    """
    # Get robust norm
    norm = get_robust_norm(family, tuning_constant)

    # Fit RLM
    model = sm.RLM(y, X, M=norm)
    result = model.fit(maxiter=max_iter, tol=tol)

    return result


def compute_robust_scale(residuals: np.ndarray, method: str = "mad") -> float:
    """
    Compute robust scale estimate.

    Parameters:
        residuals: Array of residuals
        method: Scale estimation method. Options:
                - "mad": Median Absolute Deviation (default)
                - "iqr": Interquartile Range

    Returns:
        Robust scale estimate.

    Notes:
        MAD is the most common robust scale estimator:
        MAD = median(|x - median(x)|) * 1.4826

        The constant 1.4826 makes MAD consistent with standard deviation
        for normally distributed data.

    Examples:
        >>> residuals = np.random.randn(1000)
        >>> scale = compute_robust_scale(residuals)
        >>> # scale should be close to 1.0 for standard normal
    """
    if method == "mad":
        median = np.median(residuals)
        abs_dev = np.abs(residuals - median)
        mad = np.median(abs_dev) * 1.4826
        return mad

    elif method == "iqr":
        # IQR / 1.349 is consistent with SD for normal
        q75, q25 = np.percentile(residuals, [75, 25])
        iqr_scale = (q75 - q25) / 1.349
        return iqr_scale

    else:
        raise ValueError(f"Unknown scale method '{method}'. Options: 'mad', 'iqr'")


def robust_r_squared(y: np.ndarray, fitted: np.ndarray, residuals: np.ndarray) -> float:
    """
    Compute robust R-squared.

    For robust regression, traditional R² can be misleading. This computes
    a robust version based on robust scale estimates.

    Parameters:
        y: Observed values
        fitted: Fitted values
        residuals: Residuals

    Returns:
        Robust R-squared (between 0 and 1).

    Notes:
        Formula: 1 - (MAD(residuals) / MAD(y - median(y)))²

        This measures the proportion of robust scale explained by the model.

    Examples:
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> fitted = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        >>> residuals = y - fitted
        >>> r2 = robust_r_squared(y, fitted, residuals)
    """
    # Robust scale of residuals
    mad_resid = compute_robust_scale(residuals, method="mad")

    # Robust scale of centered y
    y_centered = y - np.median(y)
    mad_y = compute_robust_scale(y_centered, method="mad")

    # Robust R²
    if mad_y == 0:
        return 1.0 if mad_resid == 0 else 0.0

    r2 = 1 - (mad_resid / mad_y) ** 2

    # Clamp to [0, 1]
    r2 = max(0.0, min(1.0, r2))

    return r2
