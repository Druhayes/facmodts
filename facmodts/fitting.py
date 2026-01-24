"""
Core fitting functions for time series factor models.

This module implements the main fit_tsfm() function and its helper functions
for fitting time series factor models using various estimation methods.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, WLS

from .control import fit_tsfm_control
from .models import TsfmControl, TsfmModel
from .utils import (
    compute_dls_weights,
    compute_excess_returns,
    make_syntactically_valid,
    remove_incomplete_cases,
    validate_column_names,
)


def fit_tsfm(
    data: Union[pl.DataFrame, pd.DataFrame, np.ndarray],
    asset_names: List[str],
    factor_names: List[str],
    mkt_name: Optional[str] = None,
    rf_name: Optional[str] = None,
    fit_method: Literal["LS", "DLS", "Robust"] = "LS",
    variable_selection: Literal["none", "stepwise", "subsets", "lars"] = "none",
    control: Optional[TsfmControl] = None,
    **kwargs,
) -> TsfmModel:
    """
    Fit a time series factor model using time series regression.

    Fits a time series (macroeconomic) factor model for one or more asset returns
    or excess returns using time series regression. Users can choose between ordinary
    least squares (LS), discounted least squares (DLS), or robust regression.
    Several variable selection options are available.

    Parameters:
        data: Time series data containing asset returns, factor returns, and optionally
              risk-free rate and market returns. Can be Polars DataFrame, pandas DataFrame,
              or numpy array. If numpy array, column names must be provided separately.
        asset_names: List of asset column names (dependent variables).
        factor_names: List of factor column names (independent variables).
        mkt_name: Name of market return column for market-timing models. Default None.
        rf_name: Name of risk-free rate column. If provided, excess returns will be
                 computed for all assets and factors. Default None.
        fit_method: Estimation method. Options:
                    - "LS": Ordinary least squares (statsmodels OLS)
                    - "DLS": Discounted least squares (weighted LS with exponential decay)
                    - "Robust": Robust regression (statsmodels RLM with M-estimators)
                    Default "LS".
        variable_selection: Variable selection method. Options:
                           - "none": Use all factors, no selection
                           - "stepwise": Stepwise regression (forward/backward/both)
                           - "subsets": Best subset selection
                           - "lars": LARS/Lasso regression
                           Default "none".
        control: TsfmControl object with control parameters. If None, created from kwargs.
        **kwargs: Additional arguments passed to fit_tsfm_control() if control is None.

    Returns:
        TsfmModel object containing:
        - asset_fit: Dictionary of fitted regression objects per asset
        - alpha: Estimated intercepts (N x 1)
        - beta: Estimated factor loadings (N x K)
        - r_squared: R-squared values per asset
        - resid_sd: Residual standard deviations per asset
        - residuals: Residual matrix (N x T)
        - data: Original Polars DataFrame
        - Additional metadata (asset_names, factor_names, fit_method, etc.)

    Raises:
        ValueError: If fit_method or variable_selection is invalid, or if required
                    columns are missing from data.
        TypeError: If data type is not supported.

    Notes:
        - Incomplete cases (rows with NA) are removed per asset to allow unequal histories.
        - Column names with spaces will be converted to periods for compatibility.
        - If rf_name is provided, excess returns are computed: R_excess = R - rf.
        - For DLS, weights decrease exponentially with decay parameter (default 0.95).
        - LARS variable selection ignores fit_method and uses its own algorithm.

    Examples:
        >>> # Basic LS regression
        >>> model = fit_tsfm(
        ...     data=returns_df,
        ...     asset_names=["Asset1", "Asset2"],
        ...     factor_names=["Factor1", "Factor2", "Factor3"],
        ...     fit_method="LS"
        ... )

        >>> # DLS with custom decay factor
        >>> model = fit_tsfm(
        ...     data=returns_df,
        ...     asset_names=["Asset1"],
        ...     factor_names=["Factor1", "Factor2"],
        ...     fit_method="DLS",
        ...     decay=0.90
        ... )

        >>> # Excess returns using risk-free rate
        >>> model = fit_tsfm(
        ...     data=returns_df,
        ...     asset_names=["Asset1", "Asset2"],
        ...     factor_names=["MKT", "SMB", "HML"],
        ...     rf_name="RF",
        ...     fit_method="LS"
        ... )

        >>> # Robust regression with custom efficiency
        >>> model = fit_tsfm(
        ...     data=returns_df,
        ...     asset_names=["Asset1"],
        ...     factor_names=["Factor1", "Factor2"],
        ...     fit_method="Robust",
        ...     efficiency=0.90
        ... )
    """
    # Validate inputs
    if fit_method not in ["LS", "DLS", "Robust"]:
        raise ValueError(f"fit_method must be 'LS', 'DLS', or 'Robust', got '{fit_method}'")

    if variable_selection not in ["none", "stepwise", "subsets", "lars"]:
        raise ValueError(
            f"variable_selection must be 'none', 'stepwise', 'subsets', or 'lars', "
            f"got '{variable_selection}'"
        )

    # Phase 1: Only support LS, DLS, and variable_selection="none"
    if fit_method == "Robust":
        raise NotImplementedError(
            "Robust regression will be implemented in Phase 2. "
            "Use fit_method='LS' or 'DLS' for now."
        )

    if variable_selection != "none":
        raise NotImplementedError(
            f"Variable selection '{variable_selection}' will be implemented in Phase 2. "
            "Use variable_selection='none' for now."
        )

    # Create control object if not provided
    if control is None:
        control = fit_tsfm_control(**kwargs)

    # Convert data to Polars DataFrame
    if isinstance(data, pd.DataFrame):
        data_pl = pl.from_pandas(data.reset_index())
    elif isinstance(data, pl.DataFrame):
        data_pl = data
    elif isinstance(data, np.ndarray):
        raise NotImplementedError("Numpy array input not yet supported. Use DataFrame.")
    else:
        raise TypeError(f"data must be DataFrame, got {type(data)}")

    # Make column names syntactically valid (spaces -> periods)
    asset_names = make_syntactically_valid(asset_names)
    factor_names = make_syntactically_valid(factor_names)
    if mkt_name is not None:
        mkt_name = make_syntactically_valid([mkt_name])[0]
    if rf_name is not None:
        rf_name = make_syntactically_valid([rf_name])[0]

    # Validate required columns exist
    required_cols = asset_names + factor_names
    if rf_name is not None:
        required_cols.append(rf_name)
    if mkt_name is not None:
        required_cols.append(mkt_name)
    validate_column_names(data_pl, required_cols, context="fit_tsfm")

    # Extract relevant columns
    dat_pl = data_pl.select(asset_names + factor_names +
                           ([rf_name] if rf_name else []) +
                           ([mkt_name] if mkt_name else []))

    # Compute excess returns if rf_name is provided
    if rf_name is not None:
        dat_pl = compute_excess_returns(dat_pl, asset_names, factor_names, rf_name)

    # Fit model based on variable selection method
    if variable_selection == "none":
        reg_list = _no_variable_selection(
            dat_pl, asset_names, factor_names, fit_method, control
        )
    elif variable_selection == "stepwise":
        # Phase 2 implementation
        raise NotImplementedError("Stepwise selection in Phase 2")
    elif variable_selection == "subsets":
        # Phase 2 implementation
        raise NotImplementedError("Subsets selection in Phase 2")
    elif variable_selection == "lars":
        # Phase 2/4 implementation
        raise NotImplementedError("LARS selection in Phase 2")
    else:
        raise ValueError(f"Unknown variable_selection: {variable_selection}")

    # Extract coefficients, statistics from fitted models
    alpha_dict = {}
    beta_dict = {}
    r2_dict = {}
    resid_sd_dict = {}
    residuals_dict = {}

    for asset in asset_names:
        if asset not in reg_list:
            continue

        fit_obj = reg_list[asset]

        # Extract coefficients
        params = fit_obj.params
        alpha_dict[asset] = params[0]  # Intercept
        beta_dict[asset] = params[1:]  # Factor loadings (already numpy array)

        # Extract R-squared
        r2_dict[asset] = fit_obj.rsquared

        # Extract residual standard deviation
        if fit_method == "DLS":
            # For DLS, compute unweighted residual SD
            resid_sd_dict[asset] = np.std(fit_obj.resid, ddof=fit_obj.df_model + 1)
        else:
            resid_sd_dict[asset] = np.sqrt(fit_obj.scale)  # statsmodels sigma

        # Extract residuals
        residuals_dict[asset] = fit_obj.resid  # Already numpy array

    # Convert dictionaries to arrays
    n_assets = len(asset_names)
    n_factors = len(factor_names)

    alpha = np.array([alpha_dict.get(asset, np.nan) for asset in asset_names])
    beta = np.array(
        [beta_dict.get(asset, np.full(n_factors, np.nan)) for asset in asset_names]
    )
    r_squared = np.array([r2_dict.get(asset, np.nan) for asset in asset_names])
    resid_sd = np.array([resid_sd_dict.get(asset, np.nan) for asset in asset_names])

    # Build residuals matrix (N x T)
    # Get maximum number of observations across all assets
    max_t = max(len(residuals_dict[asset]) for asset in residuals_dict)
    residuals = np.full((n_assets, max_t), np.nan)
    for i, asset in enumerate(asset_names):
        if asset in residuals_dict:
            resid = residuals_dict[asset]
            residuals[i, : len(resid)] = resid

    # Create TsfmModel object
    call_info = {
        "asset_names": asset_names,
        "factor_names": factor_names,
        "mkt_name": mkt_name,
        "rf_name": rf_name,
        "fit_method": fit_method,
        "variable_selection": variable_selection,
    }

    model = TsfmModel(
        asset_fit=reg_list,
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        resid_sd=resid_sd,
        residuals=residuals,
        data=dat_pl,
        asset_names=asset_names,
        factor_names=factor_names,
        fit_method=fit_method,
        variable_selection=variable_selection,
        rf_name=rf_name,
        mkt_name=mkt_name,
        call_info=call_info,
    )

    return model


def _no_variable_selection(
    dat_pl: pl.DataFrame,
    asset_names: List[str],
    factor_names: List[str],
    fit_method: Literal["LS", "DLS", "Robust"],
    control: TsfmControl,
) -> Dict[str, Any]:
    """
    Fit time series regression without variable selection.

    Fits a regression model for each asset using all specified factors.
    This function allows assets to have unequal histories by removing
    incomplete cases per asset.

    Parameters:
        dat_pl: Polars DataFrame with asset and factor returns.
        asset_names: List of asset names.
        factor_names: List of factor names.
        fit_method: "LS", "DLS", or "Robust".
        control: TsfmControl object with parameters.

    Returns:
        Dictionary mapping asset names to fitted statsmodels regression objects.

    Notes:
        - For each asset, rows with NA in asset or factor columns are removed.
        - LS: Uses statsmodels OLS.
        - DLS: Uses statsmodels WLS with exponentially decaying weights.
        - Robust: Uses statsmodels RLM (Phase 2).
    """
    reg_list = {}

    for asset in asset_names:
        # Remove incomplete cases for this asset
        reg_pl = remove_incomplete_cases(dat_pl, asset, factor_names)

        if len(reg_pl) == 0:
            # No data for this asset after removing NAs
            continue

        # Convert to pandas for statsmodels
        reg_pd = reg_pl.to_pandas()

        # Prepare X (factors) and y (asset returns)
        X = reg_pd[factor_names].values
        y = reg_pd[asset].values

        # Add constant for intercept
        X_with_const = sm.add_constant(X)

        # Fit based on method
        if fit_method == "LS":
            model = OLS(y, X_with_const)
            fit_obj = model.fit()

        elif fit_method == "DLS":
            # Compute exponentially decaying weights
            n_obs = len(reg_pd)
            weights = compute_dls_weights(n_obs, control.decay)

            model = WLS(y, X_with_const, weights=weights)
            fit_obj = model.fit()

        elif fit_method == "Robust":
            # Phase 2: Use statsmodels RLM
            raise NotImplementedError("Robust regression in Phase 2")

        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        reg_list[asset] = fit_obj

    return reg_list
