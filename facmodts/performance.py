"""
Performance attribution for factor models.

This module implements performance attribution analysis that decomposes total
returns into factor-attributed returns and specific returns.

Formula: R_t = sum(beta_k * f_kt) + u_t
where:
- R_t = total return at time t
- beta_k = exposure to factor k
- f_kt = factor k return at time t
- u_t = specific return (residual) at time t
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import polars as pl

from .models import TsfmModel


@dataclass
class PaFm:
    """
    Performance attribution results.

    Attributes
    ----------
    cum_ret_attr_f : np.ndarray
        N x K matrix of cumulative returns attributed to factors.
    cum_spec_ret : np.ndarray
        Length-N vector of cumulative specific returns.
    attr_list : dict
        Dictionary mapping asset names to DataFrames of time series attributed
        returns. Each DataFrame has columns for factor attributions and
        specific returns.
    asset_names : list of str
        Names of assets.
    factor_names : list of str
        Names of factors.
    """

    cum_ret_attr_f: np.ndarray
    cum_spec_ret: np.ndarray
    attr_list: Dict[str, pl.DataFrame]
    asset_names: List[str]
    factor_names: List[str]
    call_info: Dict = field(default_factory=dict)


def pa_fm(fit: TsfmModel) -> PaFm:
    """
    Compute cumulative mean attribution for factor models.

    Decomposes total returns into returns attributed to factors and specific
    returns using the factor model decomposition:
        R_t = sum(b_k * f_kt) + u_t

    where b_k is exposure to factor k, f_kt is factor k's return at time t,
    and u_t is the specific (residual) return.

    Parameters
    ----------
    fit : TsfmModel
        Fitted time series factor model from fit_tsfm().

    Returns
    -------
    PaFm
        Performance attribution object containing:
        - cum_ret_attr_f: Cumulative returns attributed to each factor
        - cum_spec_ret: Cumulative specific returns
        - attr_list: Time series of attributed returns for each asset

    References
    ----------
    Grinold, R. and Kahn, R. (1999) Active Portfolio Management: A Quantitative
    Approach for Producing Superior Returns and Controlling Risk. McGraw-Hill.

    Examples
    --------
    >>> import polars as pl
    >>> from facmodts import fit_tsfm, pa_fm
    >>>
    >>> # Create sample data
    >>> data = pl.DataFrame({
    ...     "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo"),
    ...     "Asset": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.02, -0.01, 0.03, 0.01, -0.01, 0.02],
    ...     "Factor1": [0.015, -0.005, 0.025, 0.008, -0.015, 0.035, 0.018, -0.008, 0.028, 0.012, -0.008, 0.015],
    ...     "Factor2": [0.01, 0.005, -0.01, 0.003, 0.008, -0.005, 0.012, -0.002, -0.008, 0.006, -0.003, 0.009]
    ... })
    >>>
    >>> # Fit model
    >>> model = fit_tsfm(
    ...     asset_names=["Asset"],
    ...     factor_names=["Factor1", "Factor2"],
    ...     data=data,
    ...     fit_method="LS"
    ... )
    >>>
    >>> # Performance attribution
    >>> pa = pa_fm(model)
    >>> print("Cumulative factor attributions:", pa.cum_ret_attr_f)
    >>> print("Cumulative specific return:", pa.cum_spec_ret)
    """
    # Validate input
    if not isinstance(fit, TsfmModel):
        raise TypeError("fit must be a TsfmModel object")

    asset_names = fit.asset_names
    factor_names = fit.factor_names
    n_assets = len(asset_names)
    n_factors = len(factor_names)

    # Initialize output arrays
    cum_ret_attr_f = np.zeros((n_assets, n_factors))
    cum_spec_ret = np.zeros(n_assets)
    attr_list = {}

    # Get data as pandas for easier manipulation
    data_pd = fit.data.to_pandas()

    # Process each asset
    for i, asset_name in enumerate(asset_names):
        # Get asset returns (handle NaNs)
        asset_col = data_pd[asset_name]
        valid_idx = ~asset_col.isna()

        # Get factor data for valid periods
        factor_data = data_pd.loc[valid_idx, factor_names].values  # T x K
        asset_returns = asset_col[valid_idx].values  # T

        # Number of valid periods
        T = len(asset_returns)

        if T == 0:
            # No valid data for this asset
            cum_ret_attr_f[i, :] = np.nan
            cum_spec_ret[i] = np.nan
            continue

        # Get beta coefficients for this asset
        beta = fit.beta[i, :]  # K

        # Compute attributed returns for each factor
        # attr_ret_k = beta_k * f_kt (T x K matrix)
        attr_ret_factors = factor_data * beta[np.newaxis, :]  # Broadcasting

        # Compute specific returns
        # Actual returns - sum of factor attributions
        fitted_returns = np.sum(attr_ret_factors, axis=1) + fit.alpha[i]
        spec_ret = asset_returns - fitted_returns  # T

        # Compute cumulative returns
        # Using compound return formula: (1+r1)*(1+r2)*...*(1+rT) - 1
        def cumulative_return(returns):
            """Compute cumulative return from series of returns."""
            return np.prod(1 + returns) - 1

        # Total cumulative return
        cum_total = cumulative_return(asset_returns)

        # Cumulative return for each factor attribution
        for k in range(n_factors):
            if not np.isnan(beta[k]):
                # Cumulative return attributed to factor k
                cum_ret_attr_f[i, k] = cumulative_return(attr_ret_factors[:, k])
            else:
                cum_ret_attr_f[i, k] = np.nan

        # Cumulative specific return
        cum_spec_ret[i] = cumulative_return(spec_ret)

        # Create time series DataFrame for this asset
        # Get dates for valid periods
        dates_idx = data_pd.index[valid_idx]

        # Build DataFrame with factor attributions and specific returns
        attr_df_data = {"date": dates_idx.to_list()}

        # Add factor attribution columns
        for k, fname in enumerate(factor_names):
            if not np.isnan(beta[k]):
                attr_df_data[fname] = attr_ret_factors[:, k]
            else:
                attr_df_data[fname] = [np.nan] * T

        # Add specific returns
        attr_df_data["specific_returns"] = spec_ret

        # Convert to Polars DataFrame
        attr_list[asset_name] = pl.DataFrame(attr_df_data)

    # Create PaFm object
    result = PaFm(
        cum_ret_attr_f=cum_ret_attr_f,
        cum_spec_ret=cum_spec_ret,
        attr_list=attr_list,
        asset_names=asset_names,
        factor_names=factor_names,
        call_info={"model_type": "tsfm"},
    )

    return result
