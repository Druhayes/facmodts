"""
Variable selection methods for time series factor models.

This module implements stepwise, subsets, and LARS variable selection methods
for factor model fitting.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from sklearn.linear_model import LassoLarsCV, LassoLarsIC
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper, WLS

from .robust import fit_rlm
from .utils import compute_dls_weights, remove_incomplete_cases


def select_stepwise(
    dat_pl: pl.DataFrame,
    asset_names: List[str],
    factor_names: List[str],
    fit_method: Literal["LS", "DLS", "Robust"],
    decay: float = 0.95,
    direction: Literal["both", "forward", "backward"] = "both",
    criterion: Literal["aic", "bic"] = "bic",
    family: str = "bisquare",
    tuning_psi: Optional[float] = None,
    max_it: int = 100,
    rel_tol: float = 1e-7,
) -> Dict[str, Any]:
    """
    Stepwise variable selection for factor models.

    Performs forward, backward, or bidirectional stepwise selection based on
    AIC or BIC criterion.

    Parameters:
        dat_pl: Polars DataFrame with asset and factor returns.
        asset_names: List of asset names.
        factor_names: List of factor names.
        fit_method: "LS", "DLS", or "Robust".
        decay: Decay factor for DLS (default 0.95).
        direction: Stepwise direction - "forward", "backward", or "both".
        criterion: Selection criterion - "aic" or "bic".
        family: Robust family for Robust method.
        tuning_psi: Tuning constant for robust method.
        max_it: Max iterations for robust method.
        rel_tol: Convergence tolerance for robust method.

    Returns:
        Dictionary mapping asset names to fitted regression objects with
        selected factors.
    """
    reg_list = {}

    for asset in asset_names:
        # Remove incomplete cases for this asset
        reg_pl = remove_incomplete_cases(dat_pl, asset, factor_names)

        if len(reg_pl) == 0:
            continue

        # Convert to pandas for statsmodels
        reg_pd = reg_pl.to_pandas()
        y = reg_pd[asset].values

        # Perform stepwise selection
        if direction == "forward":
            selected_factors = _forward_selection(
                reg_pd, y, factor_names, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )
        elif direction == "backward":
            selected_factors = _backward_elimination(
                reg_pd, y, factor_names, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )
        else:  # both
            selected_factors = _bidirectional_selection(
                reg_pd, y, factor_names, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )

        # Fit final model with selected factors
        if len(selected_factors) == 0:
            # No factors selected, fit intercept-only model
            selected_factors = []

        X = reg_pd[selected_factors].values if selected_factors else np.zeros((len(y), 0))
        X_with_const = sm.add_constant(X)

        # Fit based on method
        if fit_method == "LS":
            model = OLS(y, X_with_const)
            fit_obj = model.fit()
        elif fit_method == "DLS":
            weights = compute_dls_weights(len(reg_pd), decay)
            model = WLS(y, X_with_const, weights=weights)
            fit_obj = model.fit()
        elif fit_method == "Robust":
            fit_obj = fit_rlm(
                y, X_with_const, family=family,
                tuning_constant=tuning_psi, max_iter=max_it, tol=rel_tol
            )

        # Store selected factor names in fit object for later reference
        fit_obj.selected_factors = selected_factors
        reg_list[asset] = fit_obj

    return reg_list


def _forward_selection(
    reg_pd: pd.DataFrame,
    y: np.ndarray,
    factor_names: List[str],
    fit_method: str,
    decay: float,
    criterion: str,
    family: str,
    tuning_psi: Optional[float],
    max_it: int,
    rel_tol: float,
) -> List[str]:
    """Forward stepwise selection."""
    selected = []
    remaining = list(factor_names)

    while remaining:
        best_ic = np.inf
        best_factor = None

        for factor in remaining:
            candidate = selected + [factor]
            ic = _compute_ic(
                reg_pd, y, candidate, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )

            if ic < best_ic:
                best_ic = ic
                best_factor = factor

        # Check if adding best factor improves criterion
        if best_factor is not None:
            current_ic = _compute_ic(
                reg_pd, y, selected, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )
            if best_ic < current_ic:
                selected.append(best_factor)
                remaining.remove(best_factor)
            else:
                break
        else:
            break

    return selected


def _backward_elimination(
    reg_pd: pd.DataFrame,
    y: np.ndarray,
    factor_names: List[str],
    fit_method: str,
    decay: float,
    criterion: str,
    family: str,
    tuning_psi: Optional[float],
    max_it: int,
    rel_tol: float,
) -> List[str]:
    """Backward stepwise elimination."""
    selected = list(factor_names)

    while len(selected) > 0:
        best_ic = _compute_ic(
            reg_pd, y, selected, fit_method, decay, criterion,
            family, tuning_psi, max_it, rel_tol
        )
        worst_factor = None

        for factor in selected:
            candidate = [f for f in selected if f != factor]
            ic = _compute_ic(
                reg_pd, y, candidate, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )

            if ic < best_ic:
                best_ic = ic
                worst_factor = factor

        if worst_factor is not None:
            selected.remove(worst_factor)
        else:
            break

    return selected


def _bidirectional_selection(
    reg_pd: pd.DataFrame,
    y: np.ndarray,
    factor_names: List[str],
    fit_method: str,
    decay: float,
    criterion: str,
    family: str,
    tuning_psi: Optional[float],
    max_it: int,
    rel_tol: float,
) -> List[str]:
    """Bidirectional stepwise selection."""
    selected = []
    remaining = list(factor_names)

    while True:
        # Try adding a factor
        best_add_ic = np.inf
        best_add_factor = None

        for factor in remaining:
            candidate = selected + [factor]
            ic = _compute_ic(
                reg_pd, y, candidate, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )
            if ic < best_add_ic:
                best_add_ic = ic
                best_add_factor = factor

        # Try removing a factor
        best_remove_ic = np.inf
        best_remove_factor = None

        for factor in selected:
            candidate = [f for f in selected if f != factor]
            ic = _compute_ic(
                reg_pd, y, candidate, fit_method, decay, criterion,
                family, tuning_psi, max_it, rel_tol
            )
            if ic < best_remove_ic:
                best_remove_ic = ic
                best_remove_factor = factor

        # Current IC
        current_ic = _compute_ic(
            reg_pd, y, selected, fit_method, decay, criterion,
            family, tuning_psi, max_it, rel_tol
        )

        # Decide whether to add, remove, or stop
        if best_add_ic < current_ic and best_add_ic <= best_remove_ic:
            selected.append(best_add_factor)
            remaining.remove(best_add_factor)
        elif best_remove_ic < current_ic:
            selected.remove(best_remove_factor)
        else:
            break

    return selected


def _compute_ic(
    reg_pd: pd.DataFrame,
    y: np.ndarray,
    factor_names: List[str],
    fit_method: str,
    decay: float,
    criterion: str,
    family: str,
    tuning_psi: Optional[float],
    max_it: int,
    rel_tol: float,
) -> float:
    """Compute AIC or BIC for a given set of factors."""
    if len(factor_names) == 0:
        # Intercept-only model
        X = np.zeros((len(y), 0))
    else:
        X = reg_pd[factor_names].values

    X_with_const = sm.add_constant(X)
    n_params = X_with_const.shape[1]

    # Fit model
    if fit_method == "LS":
        model = OLS(y, X_with_const)
        result = model.fit()
    elif fit_method == "DLS":
        weights = compute_dls_weights(len(y), decay)
        model = WLS(y, X_with_const, weights=weights)
        result = model.fit()
    elif fit_method == "Robust":
        result = fit_rlm(
            y, X_with_const, family=family,
            tuning_constant=tuning_psi, max_iter=max_it, tol=rel_tol
        )

    # Compute IC
    if criterion == "aic":
        if hasattr(result, 'aic'):
            return result.aic
        else:
            # Manual AIC calculation for robust
            n = len(y)
            rss = np.sum(result.resid ** 2)
            return n * np.log(rss / n) + 2 * n_params
    else:  # bic
        if hasattr(result, 'bic'):
            return result.bic
        else:
            # Manual BIC calculation for robust
            n = len(y)
            rss = np.sum(result.resid ** 2)
            return n * np.log(rss / n) + n_params * np.log(n)


def select_all_subsets(
    dat_pl: pl.DataFrame,
    asset_names: List[str],
    factor_names: List[str],
    fit_method: Literal["LS", "DLS", "Robust"],
    decay: float = 0.95,
    nvmin: int = 1,
    nvmax: int = 8,
    family: str = "bisquare",
    tuning_psi: Optional[float] = None,
    max_it: int = 100,
    rel_tol: float = 1e-7,
) -> Dict[str, Any]:
    """
    Best subset selection for factor models.

    Evaluates all possible subsets of factors and selects the best based on BIC.

    Parameters:
        dat_pl: Polars DataFrame with asset and factor returns.
        asset_names: List of asset names.
        factor_names: List of factor names.
        fit_method: "LS", "DLS", or "Robust".
        decay: Decay factor for DLS (default 0.95).
        nvmin: Minimum number of factors to include (default 1).
        nvmax: Maximum number of factors to include (default 8).
        family: Robust family for Robust method.
        tuning_psi: Tuning constant for robust method.
        max_it: Max iterations for robust method.
        rel_tol: Convergence tolerance for robust method.

    Returns:
        Dictionary mapping asset names to fitted regression objects with
        best subset of factors.
    """
    from itertools import combinations

    reg_list = {}

    for asset in asset_names:
        # Remove incomplete cases for this asset
        reg_pl = remove_incomplete_cases(dat_pl, asset, factor_names)

        if len(reg_pl) == 0:
            continue

        # Convert to pandas for statsmodels
        reg_pd = reg_pl.to_pandas()
        y = reg_pd[asset].values

        # Try all possible subsets within size range
        best_bic = np.inf
        best_subset = []

        n_factors = len(factor_names)
        actual_nvmax = min(nvmax, n_factors)

        for size in range(nvmin, actual_nvmax + 1):
            for subset in combinations(factor_names, size):
                subset_list = list(subset)
                bic = _compute_ic(
                    reg_pd, y, subset_list, fit_method, decay, "bic",
                    family, tuning_psi, max_it, rel_tol
                )

                if bic < best_bic:
                    best_bic = bic
                    best_subset = subset_list

        # Fit final model with best subset
        if len(best_subset) == 0:
            X = np.zeros((len(y), 0))
        else:
            X = reg_pd[best_subset].values

        X_with_const = sm.add_constant(X)

        # Fit based on method
        if fit_method == "LS":
            model = OLS(y, X_with_const)
            fit_obj = model.fit()
        elif fit_method == "DLS":
            weights = compute_dls_weights(len(reg_pd), decay)
            model = WLS(y, X_with_const, weights=weights)
            fit_obj = model.fit()
        elif fit_method == "Robust":
            fit_obj = fit_rlm(
                y, X_with_const, family=family,
                tuning_constant=tuning_psi, max_iter=max_it, tol=rel_tol
            )

        # Store selected factors and BIC
        fit_obj.selected_factors = best_subset
        fit_obj.selection_bic = best_bic
        reg_list[asset] = fit_obj

    return reg_list


def select_lars(
    dat_pl: pl.DataFrame,
    asset_names: List[str],
    factor_names: List[str],
    lars_criterion: Literal["cp", "aic", "bic"] = "cp",
    cv_folds: Optional[int] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    LARS/Lasso variable selection for factor models.

    Uses LARS (Least Angle Regression) with Lasso modification to select factors.
    Model selection via Cp statistic, AIC, BIC, or cross-validation.

    Parameters:
        dat_pl: Polars DataFrame with asset and factor returns.
        asset_names: List of asset names.
        factor_names: List of factor names.
        lars_criterion: Selection criterion - "cp" (Mallows' Cp), "aic", "bic",
                        or use cv_folds for cross-validation.
                        Note: For "cp", cross-validation is used as sklearn does
                        not implement Cp directly.
        cv_folds: Number of CV folds (overrides lars_criterion if provided).

    Returns:
        Tuple of:
        - asset_fit: Dictionary of fitted LARS objects
        - alpha: Intercepts (N x 1)
        - beta: Factor loadings (N x K) - sparse, many zeros
        - r_squared: R-squared values
        - resid_sd: Residual standard deviations
    """
    asset_fit = {}
    alpha = np.full(len(asset_names), np.nan)
    beta = np.full((len(asset_names), len(factor_names)), np.nan)
    r_squared = np.full(len(asset_names), np.nan)
    resid_sd = np.full(len(asset_names), np.nan)

    for i, asset in enumerate(asset_names):
        # Remove incomplete cases for this asset
        reg_pl = remove_incomplete_cases(dat_pl, asset, factor_names)

        if len(reg_pl) == 0:
            continue

        # Convert to pandas then numpy
        reg_pd = reg_pl.to_pandas()
        X = reg_pd[factor_names].values
        y = reg_pd[asset].values

        # Fit LARS model
        if cv_folds is not None:
            # Use cross-validation
            lars_model = LassoLarsCV(cv=cv_folds, fit_intercept=True)
        elif lars_criterion == "cp":
            # sklearn doesn't support Cp, use CV as proxy
            # Cp minimization is similar to CV minimization
            lars_model = LassoLarsCV(cv=10, fit_intercept=True)
        else:
            # Use information criterion (aic or bic)
            lars_model = LassoLarsIC(criterion=lars_criterion, fit_intercept=True)

        lars_model.fit(X, y)

        # Extract coefficients
        alpha[i] = lars_model.intercept_
        beta[i, :] = lars_model.coef_

        # Compute fitted values and residuals
        y_pred = lars_model.predict(X)
        residuals = y - y_pred

        # Compute R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared[i] = 1 - (ss_res / ss_tot)

        # Compute residual SD
        n_nonzero = np.sum(lars_model.coef_ != 0)
        df_resid = len(y) - n_nonzero - 1  # -1 for intercept
        resid_sd[i] = np.sqrt(ss_res / df_resid) if df_resid > 0 else np.nan

        # Store model object
        asset_fit[asset] = lars_model

    return asset_fit, alpha, beta, r_squared, resid_sd
