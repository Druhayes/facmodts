"""
Convenience wrapper functions for specialized time series factor models.

This module provides wrapper functions that simplify common factor model configurations:
- fit_tsfm_mt: Market-timing models (Henriksson-Merton style)
- fit_tsfm_up_dn: Separate up/down market models
- fit_ff3_model: Fama-French 3-factor model
- fit_ff4_model: Fama-French 4-factor model
"""

from typing import List, Literal, Optional

import numpy as np
import polars as pl

from .control import fit_tsfm_control
from .fitting import fit_tsfm
from .models import TsfmControl, TsfmModel, TsfmUpDn


def fit_tsfm_mt(
    asset_names: List[str],
    mkt_name: str,
    rf_name: Optional[str] = None,
    data: pl.DataFrame = None,
    fit_method: Literal["LS", "DLS", "Robust"] = "LS",
    control: Optional[TsfmControl] = None,
    **kwargs
) -> TsfmModel:
    """
    Fit a market-timing time series factor model.

    This wrapper adds a market-timing factor to capture managers' timing skills
    following Henriksson & Merton (1981). The timing factor is constructed as:
        down_market = max(0, R_f - R_m)

    The coefficient of this down-market factor can be interpreted as the number
    of "free" put options on the market provided by market-timing skills.

    Parameters
    ----------
    asset_names : list of str
        Names of assets whose returns are the dependent variables.
    mkt_name : str
        Name of the market return column (required).
    rf_name : str, optional
        Name of risk-free rate column. If provided, excess returns are computed
        for all assets and the market. Default is None.
    data : polars.DataFrame
        Time series data containing asset returns, market returns, and optionally
        risk-free rate.
    fit_method : {"LS", "DLS", "Robust"}, default "LS"
        Estimation method: least squares, discounted least squares, or robust.
    control : TsfmControl, optional
        Control parameters for fit_tsfm. If None, constructed from **kwargs.
    **kwargs
        Additional arguments passed to fit_tsfm_control().

    Returns
    -------
    TsfmModel
        Fitted time series factor model with 2 factors:
        - mkt_name: Market (excess) returns
        - "down_market": max(0, R_f - R_m) timing factor

    References
    ----------
    Henriksson, R. D., & Merton, R. C. (1981). On market timing and investment
    performance. II. Statistical procedures for evaluating forecasting skills.
    Journal of Business, 513-533.

    Christopherson, J. A., Carino, D. R., & Ferson, W. E. (2009). Portfolio
    performance measurement and benchmarking. McGraw Hill Professional. pp.127-133

    Examples
    --------
    >>> import polars as pl
    >>> from facmodts import fit_tsfm_mt
    >>>
    >>> # Create sample data
    >>> data = pl.DataFrame({
    ...     "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo"),
    ...     "Fund": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.02, -0.01, 0.03, 0.01, -0.01, 0.02],
    ...     "Market": [0.015, -0.005, 0.025, 0.008, -0.015, 0.035, 0.018, -0.008, 0.028, 0.012, -0.008, 0.015],
    ...     "RF": [0.001] * 12
    ... })
    >>>
    >>> # Fit market-timing model
    >>> mt_model = fit_tsfm_mt(
    ...     asset_names=["Fund"],
    ...     mkt_name="Market",
    ...     rf_name="RF",
    ...     data=data,
    ...     fit_method="LS"
    ... )
    >>>
    >>> # Inspect timing coefficient (beta[0, 1] is the "down_market" loading)
    >>> print(f"Timing coefficient: {mt_model.beta[0, 1]:.4f}")
    """
    if mkt_name is None:
        raise ValueError("mkt_name is required for market-timing models")

    # Create control if not provided
    if control is None:
        control = fit_tsfm_control(**kwargs)

    # Extract relevant columns
    cols_to_select = asset_names + [mkt_name]
    if rf_name is not None:
        cols_to_select.append(rf_name)

    dat = data.select(cols_to_select)

    # Compute excess returns if rf_name provided
    if rf_name is not None:
        # Convert to excess returns
        rf_series = dat[rf_name]
        for col in asset_names + [mkt_name]:
            dat = dat.with_columns((pl.col(col) - rf_series).alias(col))
        # Drop RF column after computing excess returns
        dat = dat.drop(rf_name)

    # Create market-timing factor: down_market = max(0, R_f - R_m)
    # Since we already have excess returns, this is max(0, -R_m)
    # Or equivalently: max(0, 0 - R_m) where R_m is excess market return
    down_market = dat[mkt_name].to_numpy().copy()
    down_market[down_market > 0] = 0  # Keep only negative values
    down_market = -down_market  # Flip sign: max(0, -R_m)

    # Add down_market as a new column
    dat = dat.with_columns(pl.Series("down_market", down_market))

    # Factor names: market and down_market
    factor_names = [mkt_name, "down_market"]

    # Fit model using fitTsfm
    model = fit_tsfm(
        asset_names=asset_names,
        factor_names=factor_names,
        rf_name=None,  # Already computed excess returns
        data=dat,
        fit_method=fit_method,
        variable_selection="none",
        control=control,
    )

    # Add mkt_name to model for consistency with R
    model.mkt_name = mkt_name
    model.call_info["wrapper"] = "fit_tsfm_mt"

    return model


def fit_tsfm_up_dn(
    asset_names: List[str],
    mkt_name: str,
    rf_name: Optional[str] = None,
    data: pl.DataFrame = None,
    fit_method: Literal["LS", "DLS", "Robust"] = "LS",
    control: Optional[TsfmControl] = None,
    **kwargs
) -> TsfmUpDn:
    """
    Fit separate time series factor models for up and down markets.

    This wrapper fits two separate models to capture different factor sensitivities
    in bull vs bear markets. The market is split based on the sign of (excess)
    market returns: up market periods have R_m >= 0, down market have R_m < 0.

    Parameters
    ----------
    asset_names : list of str
        Names of assets whose returns are the dependent variables.
    mkt_name : str
        Name of the market return column used for up/down classification (required).
    rf_name : str, optional
        Name of risk-free rate column. If provided, excess returns are computed.
        Default is None.
    data : polars.DataFrame
        Time series data containing asset returns, market returns, and optionally
        risk-free rate.
    fit_method : {"LS", "DLS", "Robust"}, default "LS"
        Estimation method for both models.
    control : TsfmControl, optional
        Control parameters for fit_tsfm. If None, constructed from **kwargs.
    **kwargs
        Additional arguments passed to fit_tsfm_control().

    Returns
    -------
    TsfmUpDn
        Object containing:
        - up_model: TsfmModel fitted on up market periods (R_m >= 0)
        - down_model: TsfmModel fitted on down market periods (R_m < 0)
        - up_market_flag: Boolean array indicating up market periods
        - market_name: Name of market column used for classification
        - call_info: Original function call information

    References
    ----------
    Christopherson, J. A., Carino, D. R., & Ferson, W. E. (2009). Portfolio
    performance measurement and benchmarking. McGraw Hill Professional.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from facmodts import fit_tsfm_up_dn
    >>>
    >>> # Create sample data with clear up/down markets
    >>> np.random.seed(42)
    >>> n = 60
    >>> mkt = np.random.randn(n) * 0.02
    >>> data = pl.DataFrame({
    ...     "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1), "1d")[:n],
    ...     "Asset1": 0.5 * mkt + np.random.randn(n) * 0.01,
    ...     "Asset2": 0.8 * mkt + np.random.randn(n) * 0.015,
    ...     "Market": mkt
    ... })
    >>>
    >>> # Fit up/down model
    >>> updn_model = fit_tsfm_up_dn(
    ...     asset_names=["Asset1", "Asset2"],
    ...     mkt_name="Market",
    ...     data=data,
    ...     fit_method="LS"
    ... )
    >>>
    >>> # Compare betas in up vs down markets
    >>> print("Up market betas:", updn_model.up_model.beta[:, 0])
    >>> print("Down market betas:", updn_model.down_model.beta[:, 0])
    """
    if mkt_name is None:
        raise ValueError("mkt_name is required for up/down market models")

    # Create control if not provided
    if control is None:
        control = fit_tsfm_control(**kwargs)

    # Extract relevant columns
    cols_to_select = asset_names + [mkt_name]
    if rf_name is not None:
        cols_to_select.append(rf_name)

    dat = data.select(cols_to_select)

    # Compute excess returns if rf_name provided
    if rf_name is not None:
        rf_series = dat[rf_name]
        for col in asset_names + [mkt_name]:
            dat = dat.with_columns((pl.col(col) - rf_series).alias(col))
        # Drop RF column
        dat = dat.drop(rf_name)

    # Classify up vs down markets
    mkt_returns = dat[mkt_name].to_numpy()
    up_market_flag = mkt_returns >= 0

    # Split data into up and down markets
    dat_up = dat.filter(pl.Series(up_market_flag))
    dat_down = dat.filter(pl.Series(~up_market_flag))

    # Fit up market model
    model_up = fit_tsfm(
        asset_names=asset_names,
        factor_names=[mkt_name],
        rf_name=None,  # Already computed excess returns
        data=dat_up,
        fit_method=fit_method,
        variable_selection="none",
        control=control,
    )
    model_up.call_info["market_regime"] = "up"

    # Fit down market model
    model_down = fit_tsfm(
        asset_names=asset_names,
        factor_names=[mkt_name],
        rf_name=None,
        data=dat_down,
        fit_method=fit_method,
        variable_selection="none",
        control=control,
    )
    model_down.call_info["market_regime"] = "down"

    # Create TsfmUpDn object
    result = TsfmUpDn(
        up_model=model_up,
        down_model=model_down,
        up_market_flag=up_market_flag,
        market_name=mkt_name,
        call_info={
            "asset_names": asset_names,
            "mkt_name": mkt_name,
            "rf_name": rf_name,
            "fit_method": fit_method,
            "wrapper": "fit_tsfm_up_dn",
        },
    )

    return result


def fit_ff3_model(
    asset_names: List[str],
    factor_data: pl.DataFrame,
    asset_data: pl.DataFrame = None,
    fit_method: Literal["LS", "Robust"] = "LS",
    control: Optional[TsfmControl] = None,
    **kwargs
) -> TsfmModel:
    """
    Fit Fama-French 3-factor model.

    Convenience wrapper for fitting the Fama-French 3-factor model consisting of:
    - MKT: Market excess returns (R_m - R_f)
    - SMB: Small-minus-big (size factor)
    - HML: High-minus-low (value factor)

    Parameters
    ----------
    asset_names : list of str
        Names of assets to fit the model for.
    factor_data : polars.DataFrame
        DataFrame containing FF3 factors. Must have columns: "MKT", "SMB", "HML".
        Should also have a date/time column for merging with asset_data.
    asset_data : polars.DataFrame, optional
        DataFrame containing asset returns. If None, factor_data must also contain
        asset columns.
    fit_method : {"LS", "Robust"}, default "LS"
        Estimation method.
    control : TsfmControl, optional
        Control parameters. If None, constructed from **kwargs.
    **kwargs
        Additional arguments passed to fit_tsfm_control().

    Returns
    -------
    TsfmModel
        Fitted FF3 model with factors: MKT, SMB, HML.

    Notes
    -----
    The Fama-French factors are typically already expressed as excess returns,
    so no additional risk-free rate adjustment is needed.

    References
    ----------
    Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on
    stocks and bonds. Journal of Financial Economics, 33(1), 3-56.

    Examples
    --------
    >>> import polars as pl
    >>> from facmodts import fit_ff3_model
    >>>
    >>> # Load FF3 factor data (example)
    >>> ff3_data = pl.DataFrame({
    ...     "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo"),
    ...     "MKT": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.02, -0.01, 0.03, 0.01, -0.01, 0.02],
    ...     "SMB": [0.01, 0.005, -0.01, 0.008, 0.002, -0.005, 0.01, 0.003, -0.008, 0.005, 0.001, -0.003],
    ...     "HML": [-0.005, 0.01, 0.002, -0.008, 0.015, 0.001, -0.01, 0.012, -0.003, 0.008, -0.005, 0.004],
    ...     "Fund": [0.025, -0.005, 0.035, 0.015, -0.015, 0.045, 0.025, -0.005, 0.035, 0.015, -0.005, 0.025]
    ... })
    >>>
    >>> # Fit FF3 model
    >>> ff3_model = fit_ff3_model(
    ...     asset_names=["Fund"],
    ...     factor_data=ff3_data,
    ...     fit_method="LS"
    ... )
    >>>
    >>> print("FF3 loadings:", ff3_model.beta[0, :])
    """
    # Check that factor_data has required columns
    required_factors = ["MKT", "SMB", "HML"]
    missing = [f for f in required_factors if f not in factor_data.columns]
    if missing:
        raise ValueError(f"factor_data missing required columns: {missing}")

    # Merge asset and factor data if separate
    if asset_data is not None:
        # Assume first column is date/index
        merge_key = factor_data.columns[0]
        data = asset_data.join(factor_data, on=merge_key, how="inner")
    else:
        # Factors and assets in same DataFrame
        data = factor_data

    # Create control if not provided
    if control is None:
        control = fit_tsfm_control(**kwargs)

    # Fit model
    model = fit_tsfm(
        asset_names=asset_names,
        factor_names=required_factors,
        rf_name=None,  # FF factors already excess returns
        data=data,
        fit_method=fit_method,
        variable_selection="none",
        control=control,
    )

    model.call_info["wrapper"] = "fit_ff3_model"

    return model


def fit_ff4_model(
    asset_names: List[str],
    factor_data: pl.DataFrame,
    asset_data: pl.DataFrame = None,
    fit_method: Literal["LS", "Robust"] = "LS",
    control: Optional[TsfmControl] = None,
    **kwargs
) -> TsfmModel:
    """
    Fit Fama-French 4-factor model (Carhart model).

    Convenience wrapper for fitting the Fama-French 4-factor model, which extends
    FF3 by adding the momentum factor:
    - MKT: Market excess returns (R_m - R_f)
    - SMB: Small-minus-big (size factor)
    - HML: High-minus-low (value factor)
    - MOM (or UMD): Winners-minus-losers (momentum factor)

    Parameters
    ----------
    asset_names : list of str
        Names of assets to fit the model for.
    factor_data : polars.DataFrame
        DataFrame containing FF4 factors. Must have columns: "MKT", "SMB", "HML",
        and either "MOM" or "UMD". Should also have a date/time column.
    asset_data : polars.DataFrame, optional
        DataFrame containing asset returns. If None, factor_data must also contain
        asset columns.
    fit_method : {"LS", "Robust"}, default "LS"
        Estimation method.
    control : TsfmControl, optional
        Control parameters. If None, constructed from **kwargs.
    **kwargs
        Additional arguments passed to fit_tsfm_control().

    Returns
    -------
    TsfmModel
        Fitted FF4 model with factors: MKT, SMB, HML, MOM/UMD.

    Notes
    -----
    The Fama-French factors are typically already expressed as excess returns.
    The momentum factor may be named "MOM" or "UMD" (up-minus-down) depending
    on the data source.

    References
    ----------
    Carhart, M. M. (1997). On persistence in mutual fund performance. Journal of
    Finance, 52(1), 57-82.

    Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on
    stocks and bonds. Journal of Financial Economics, 33(1), 3-56.

    Examples
    --------
    >>> import polars as pl
    >>> from facmodts import fit_ff4_model
    >>>
    >>> # Load FF4 factor data
    >>> ff4_data = pl.DataFrame({
    ...     "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo"),
    ...     "MKT": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.02, -0.01, 0.03, 0.01, -0.01, 0.02],
    ...     "SMB": [0.01, 0.005, -0.01, 0.008, 0.002, -0.005, 0.01, 0.003, -0.008, 0.005, 0.001, -0.003],
    ...     "HML": [-0.005, 0.01, 0.002, -0.008, 0.015, 0.001, -0.01, 0.012, -0.003, 0.008, -0.005, 0.004],
    ...     "MOM": [0.008, -0.003, 0.012, -0.005, 0.01, -0.002, 0.008, -0.004, 0.011, -0.003, 0.007, -0.001],
    ...     "Fund": [0.025, -0.005, 0.035, 0.015, -0.015, 0.045, 0.025, -0.005, 0.035, 0.015, -0.005, 0.025]
    ... })
    >>>
    >>> # Fit FF4 model
    >>> ff4_model = fit_ff4_model(
    ...     asset_names=["Fund"],
    ...     factor_data=ff4_data,
    ...     fit_method="LS"
    ... )
    >>>
    >>> print("FF4 loadings:", ff4_model.beta[0, :])
    """
    # Check for MOM or UMD
    mom_factor = None
    if "MOM" in factor_data.columns:
        mom_factor = "MOM"
    elif "UMD" in factor_data.columns:
        mom_factor = "UMD"
    else:
        raise ValueError("factor_data must contain 'MOM' or 'UMD' momentum factor")

    # Check other required factors
    required_factors = ["MKT", "SMB", "HML", mom_factor]
    missing = [f for f in required_factors if f not in factor_data.columns]
    if missing:
        raise ValueError(f"factor_data missing required columns: {missing}")

    # Merge asset and factor data if separate
    if asset_data is not None:
        merge_key = factor_data.columns[0]
        data = asset_data.join(factor_data, on=merge_key, how="inner")
    else:
        data = factor_data

    # Create control if not provided
    if control is None:
        control = fit_tsfm_control(**kwargs)

    # Fit model
    model = fit_tsfm(
        asset_names=asset_names,
        factor_names=required_factors,
        rf_name=None,  # FF factors already excess returns
        data=data,
        fit_method=fit_method,
        variable_selection="none",
        control=control,
    )

    model.call_info["wrapper"] = "fit_ff4_model"

    return model
