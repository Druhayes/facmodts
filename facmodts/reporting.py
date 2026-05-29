"""
Reporting methods for factor models.

This module provides summary, print, and predict methods for fitted factor models
and performance attribution results.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
import pandas as pd
from scipy import stats

from .models import TsfmModel, TsfmUpDn
from .performance import PaFm


def summary_tsfm(model: TsfmModel, se_type: str = "default") -> Dict:
    """
    Summarize a fitted time series factor model.

    Computes coefficient summaries with standard errors and t-statistics
    for each asset in the factor model.

    Parameters
    ----------
    model : TsfmModel
        Fitted time series factor model.
    se_type : str, default "default"
        Standard error type: "default" for homoskedastic, "HC" for
        heteroskedasticity-consistent (White's), or "HAC" for
        heteroskedasticity-autocorrelation-consistent (Newey-West).
        Note: Only "default" is currently fully implemented.

    Returns
    -------
    dict
        Summary dictionary containing:
        - call_info: Original model call information
        - se_type: Standard error type used
        - summaries: List of per-asset summaries with coefficients, std errors,
          t-stats, p-values, R², and residual SD

    Examples
    --------
    >>> from facmodts import fit_tsfm, summary_tsfm
    >>> # ... fit model ...
    >>> summary = summary_tsfm(model)
    >>> print_summary_tsfm(summary)
    """
    if se_type not in ["default", "HC", "HAC"]:
        raise ValueError("se_type must be 'default', 'HC', or 'HAC'")

    if model.fit_method == "Robust" and se_type != "default":
        raise ValueError(
            "HC/HAC standard errors only applicable for LS/DLS methods"
        )

    summaries = []

    for i, asset_name in enumerate(model.asset_names):
        # Get fitted object for this asset
        fit_obj = model.asset_fit[asset_name]

        # Extract coefficients
        alpha = model.alpha[i]
        beta = model.beta[i, :]
        coefs = np.concatenate([[alpha], beta])

        # Build coefficient names
        coef_names = ["(Intercept)"] + model.factor_names

        # For LARS, standard errors not available
        if model.variable_selection == "lars":
            summary = {
                "asset": asset_name,
                "coefficients": coefs,
                "coef_names": coef_names,
                "std_errors": None,
                "t_stats": None,
                "p_values": None,
                "r_squared": model.r_squared[i],
                "resid_sd": model.resid_sd[i],
                "method": "LARS (std errors not available)",
            }
        else:
            # For LS/DLS/Robust: extract from statsmodels fit object
            if hasattr(fit_obj, "summary"):
                # Statsmodels object
                params = fit_obj.params
                std_errors = fit_obj.bse
                t_stats = fit_obj.tvalues
                p_values = fit_obj.pvalues

                # Convert to numpy if pandas Series
                if hasattr(params, "values"):
                    param_vals = params.values
                    coef_names_list = params.index.tolist()
                    std_err_vals = std_errors.values
                    t_stat_vals = t_stats.values
                    p_val_vals = p_values.values
                else:
                    # Already numpy arrays
                    param_vals = params
                    coef_names_list = coef_names  # Use our constructed names
                    std_err_vals = std_errors
                    t_stat_vals = t_stats
                    p_val_vals = p_values

                summary = {
                    "asset": asset_name,
                    "coefficients": param_vals,
                    "coef_names": coef_names_list,
                    "std_errors": std_err_vals,
                    "t_stats": t_stat_vals,
                    "p_values": p_val_vals,
                    "r_squared": model.r_squared[i],
                    "resid_sd": model.resid_sd[i],
                    "method": model.fit_method,
                }
            else:
                # Fallback for other fit types
                summary = {
                    "asset": asset_name,
                    "coefficients": coefs,
                    "coef_names": coef_names,
                    "std_errors": None,
                    "t_stats": None,
                    "p_values": None,
                    "r_squared": model.r_squared[i],
                    "resid_sd": model.resid_sd[i],
                    "method": model.fit_method,
                }

        summaries.append(summary)

    return {
        "call_info": model.call_info,
        "se_type": se_type,
        "summaries": summaries,
    }


def print_summary_tsfm(summary: Dict, digits: int = 3):
    """
    Print summary of fitted time series factor model.

    Parameters
    ----------
    summary : dict
        Summary dictionary from summary_tsfm().
    digits : int, default 3
        Number of decimal places to print.
    """
    print("\nTime Series Factor Model Summary")
    print("=" * 70)

    if summary["call_info"]:
        print("\nCall Information:")
        for key, value in summary["call_info"].items():
            if key not in ["data"]:  # Skip data
                print(f"  {key}: {value}")

    print(f"\nStandard Error Type: {summary['se_type']}")

    for asset_summary in summary["summaries"]:
        print("\n" + "-" * 70)
        print(f"Asset: {asset_summary['asset']}")
        print(f"Method: {asset_summary['method']}")
        print("-" * 70)

        # Print coefficients table
        coefs = asset_summary["coefficients"]
        names = asset_summary["coef_names"]

        if asset_summary["std_errors"] is not None:
            # Full table with std errors
            print(f"\n{'Coefficient':<15} {'Estimate':>12} {'Std Error':>12} "
                  f"{'t-stat':>10} {'Pr(>|t|)':>10}")
            print("-" * 70)

            for j, name in enumerate(names):
                est = coefs[j]
                se = asset_summary["std_errors"][j]
                t = asset_summary["t_stats"][j]
                p = asset_summary["p_values"][j]

                # Significance stars
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                elif p < 0.1:
                    sig = "."
                else:
                    sig = ""

                print(f"{name:<15} {est:>12.{digits}f} {se:>12.{digits}f} "
                      f"{t:>10.{digits}f} {p:>10.{digits}f} {sig}")

            print("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        else:
            # Just coefficients (LARS)
            print(f"\n{'Coefficient':<15} {'Estimate':>12}")
            print("-" * 30)
            for j, name in enumerate(names):
                if not np.isnan(coefs[j]) and coefs[j] != 0:
                    print(f"{name:<15} {coefs[j]:>12.{digits}f}")
                else:
                    print(f"{name:<15} {'.':<12}")

        print(f"\nR-squared: {asset_summary['r_squared']:.{digits}f}")
        print(f"Residual SD: {asset_summary['resid_sd']:.{digits}f}")

    print("\n" + "=" * 70)


def print_tsfm(model: TsfmModel, digits: int = 3):
    """
    Print a brief summary of fitted time series factor model.

    Parameters
    ----------
    model : TsfmModel
        Fitted time series factor model.
    digits : int, default 3
        Number of decimal places.
    """
    print("\nTime Series Factor Model")
    print("=" * 60)

    if model.call_info:
        print(f"\nFit method: {model.fit_method}")
        print(f"Variable selection: {model.variable_selection}")

    # Model dimensions
    n_assets = len(model.asset_names)
    n_factors = len(model.factor_names)
    n_periods = model.data.shape[0]

    print(f"\nDimensions:")
    print(f"  Assets:  {n_assets}")
    print(f"  Factors: {n_factors}")
    print(f"  Periods: {n_periods}")

    # Coefficients
    print(f"\nRegression Alphas:")
    for i, name in enumerate(model.asset_names):
        print(f"  {name:<20}: {model.alpha[i]:>10.{digits}f}")

    print(f"\nFactor Betas:")
    # Print header
    print(f"  {'Asset':<20}", end="")
    for fname in model.factor_names:
        print(f"{fname:>12}", end="")
    print()

    # Print betas
    for i, name in enumerate(model.asset_names):
        print(f"  {name:<20}", end="")
        for j in range(n_factors):
            beta_val = model.beta[i, j]
            if np.isnan(beta_val) or (model.variable_selection == "lars" and beta_val == 0):
                print(f"{'.':<12}", end="")
            else:
                print(f"{beta_val:>12.{digits}f}", end="")
        print()

    print(f"\nR-squared:")
    for i, name in enumerate(model.asset_names):
        print(f"  {name:<20}: {model.r_squared[i]:>10.{digits}f}")

    print(f"\nResidual SD:")
    for i, name in enumerate(model.asset_names):
        print(f"  {name:<20}: {model.resid_sd[i]:>10.{digits}f}")

    print("=" * 60)


def predict_tsfm(
    model: TsfmModel,
    newdata: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
) -> np.ndarray:
    """
    Predict asset returns based on fitted time series factor model.

    Uses fitted factor loadings and new factor data to predict asset returns:
        predicted_return = alpha + sum(beta_k * factor_k)

    Parameters
    ----------
    model : TsfmModel
        Fitted time series factor model.
    newdata : DataFrame, optional
        New factor data for prediction. Must contain columns matching
        model.factor_names. If None, uses in-sample fitted values.

    Returns
    -------
    np.ndarray
        Matrix of predicted returns (N assets x T periods), or dictionary
        mapping asset names to prediction arrays if assets have unequal history.

    Examples
    --------
    >>> from facmodts import fit_tsfm, predict_tsfm
    >>> import polars as pl
    >>>
    >>> # Fit model
    >>> # ... model fitting code ...
    >>>
    >>> # In-sample predictions
    >>> fitted = predict_tsfm(model)
    >>>
    >>> # Out-of-sample predictions
    >>> new_factors = pl.DataFrame({
    ...     "Factor1": [0.01, 0.02, -0.01],
    ...     "Factor2": [0.005, -0.005, 0.01]
    ... })
    >>> predictions = predict_tsfm(model, newdata=new_factors)
    """
    if newdata is None:
        # Use in-sample fitted values
        # For each asset, compute alpha + beta * factors
        n_assets = len(model.asset_names)
        n_periods = model.data.shape[0]

        # Get factor data
        factor_data = model.data.select(model.factor_names).to_numpy()  # T x K

        # Compute predictions for each asset
        predictions = {}
        for i, asset_name in enumerate(model.asset_names):
            alpha = model.alpha[i]
            beta = model.beta[i, :]  # K

            # Handle NaN betas (from variable selection)
            beta_clean = np.where(np.isnan(beta), 0, beta)

            # Prediction: alpha + beta * factors
            pred = alpha + factor_data @ beta_clean  # T

            predictions[asset_name] = pred

        # Check if all have same length
        lengths = [len(p) for p in predictions.values()]
        if len(set(lengths)) == 1:
            # All same length, return as matrix
            return np.column_stack([predictions[name] for name in model.asset_names]).T
        else:
            # Unequal lengths, return dict
            return predictions
    else:
        # Out-of-sample prediction
        if isinstance(newdata, pd.DataFrame):
            newdata = pl.from_pandas(newdata)

        # Check that newdata has required factor columns
        missing = [f for f in model.factor_names if f not in newdata.columns]
        if missing:
            raise ValueError(f"newdata missing required factors: {missing}")

        # Get factor data
        factor_data = newdata.select(model.factor_names).to_numpy()  # T_new x K
        n_new = factor_data.shape[0]
        n_assets = len(model.asset_names)

        # Compute predictions
        predictions = np.zeros((n_assets, n_new))

        for i in range(n_assets):
            alpha = model.alpha[i]
            beta = model.beta[i, :]

            # Handle NaN betas
            beta_clean = np.where(np.isnan(beta), 0, beta)

            # Prediction
            predictions[i, :] = alpha + factor_data @ beta_clean

        return predictions


def print_pafm(pa: PaFm, digits: int = 3):
    """
    Print performance attribution results.

    Parameters
    ----------
    pa : PaFm
        Performance attribution object.
    digits : int, default 3
        Number of decimal places.
    """
    print("\nPerformance Attribution (Factor Model)")
    print("=" * 70)

    print("\nMean of returns attributed to factors:")
    print(f"\n{'Asset':<20}", end="")
    for fname in pa.factor_names:
        print(f"{fname:>15}", end="")
    print(f"{'Specific':>15}")

    for i, asset_name in enumerate(pa.asset_names):
        print(f"{asset_name:<20}", end="")

        # Get time series for this asset
        attr_ts = pa.attr_list[asset_name]

        # Compute means
        for fname in pa.factor_names:
            mean_val = attr_ts[fname].mean()
            print(f"{mean_val:>15.{digits}f}", end="")

        # Specific returns mean
        spec_mean = attr_ts["specific_returns"].mean()
        print(f"{spec_mean:>15.{digits}f}")

    print("\nCumulative returns attributed to factors:")
    print(f"\n{'Asset':<20}", end="")
    for fname in pa.factor_names:
        print(f"{fname:>15}", end="")
    print(f"{'Specific':>15}")

    for i, asset_name in enumerate(pa.asset_names):
        print(f"{asset_name:<20}", end="")

        for j in range(len(pa.factor_names)):
            cum_val = pa.cum_ret_attr_f[i, j]
            print(f"{cum_val:>15.{digits}f}", end="")

        # Cumulative specific return
        cum_spec = pa.cum_spec_ret[i]
        print(f"{cum_spec:>15.{digits}f}")

    print("=" * 70)


def summary_pafm(pa: PaFm, digits: int = 3):
    """
    Summarize performance attribution results.

    Parameters
    ----------
    pa : PaFm
        Performance attribution object.
    digits : int, default 3
        Number of decimal places.
    """
    print("\nPerformance Attribution Summary")
    print("=" * 70)

    print("\nMean of returns attributed to factors:")
    print(f"\n{'Asset':<20}", end="")
    for fname in pa.factor_names:
        print(f"{fname:>15}", end="")
    print(f"{'Specific':>15}")

    for i, asset_name in enumerate(pa.asset_names):
        print(f"{asset_name:<20}", end="")

        attr_ts = pa.attr_list[asset_name]

        for fname in pa.factor_names:
            mean_val = attr_ts[fname].mean()
            print(f"{mean_val:>15.{digits}f}", end="")

        spec_mean = attr_ts["specific_returns"].mean()
        print(f"{spec_mean:>15.{digits}f}")

    print("\n\nStandard Deviation of returns attributed to factors:")
    print(f"\n{'Asset':<20}", end="")
    for fname in pa.factor_names:
        print(f"{fname:>15}", end="")
    print(f"{'Specific':>15}")

    for i, asset_name in enumerate(pa.asset_names):
        print(f"{asset_name:<20}", end="")

        attr_ts = pa.attr_list[asset_name]

        for fname in pa.factor_names:
            std_val = attr_ts[fname].std()
            print(f"{std_val:>15.{digits}f}", end="")

        spec_std = attr_ts["specific_returns"].std()
        print(f"{spec_std:>15.{digits}f}")

    print("=" * 70)
