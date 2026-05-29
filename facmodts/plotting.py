"""
Plotting methods for factor models.

This module provides visualization functions for fitted factor models,
performance attribution, and up/down market models.
"""

from typing import List, Optional, Literal, Union
import warnings

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy import stats

from .models import TsfmModel, TsfmUpDn
from .performance import PaFm
from .decomposition import fm_sd_decomp, fm_var_decomp, fm_es_decomp


def plot_tsfm(
    model: TsfmModel,
    which: Optional[Union[int, List[int]]] = None,
    asset_name: Optional[str] = None,
    plot_single: bool = False,
    a_sub: Optional[Union[List[int], List[str]]] = None,
    f_sub: Optional[Union[List[int], List[str]]] = None,
    colorset: Optional[List[str]] = None,
    legend_loc: str = "best",
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot diagnostics for fitted time series factor model.

    Parameters
    ----------
    model : TsfmModel
        Fitted time series factor model.
    which : int or list of int, optional
        Plot selection:

        **Individual asset plots (plot_single=True):**
        1 = Actual and fitted time series
        2 = Actual vs fitted scatter
        3 = Residuals vs fitted
        4 = Residuals with SE bands
        5 = Squared residuals time series
        6 = Absolute residuals time series
        7 = ACF/PACF of residuals
        8 = ACF/PACF of squared residuals
        9 = Histogram of residuals
        10 = QQ-plot of residuals

        **Group plots (plot_single=False):**
        1 = Alpha coefficients
        2 = Beta coefficients
        3 = Actual and fitted (multiple assets)
        4 = R-squared
        5 = Residual volatility
        6 = Residual correlation
        7 = Factor contribution to SD
        8 = Factor contribution to VaR
        9 = Factor contribution to ES

    asset_name : str, optional
        Asset name for individual plots (required if plot_single=True and
        multiple assets in model).
    plot_single : bool, default False
        If True, create individual asset plots. If False, create group plots.
    a_sub : list of int or str, optional
        Subset of asset indices/names for group plots. Default is first 6 assets.
    f_sub : list of int or str, optional
        Subset of factor indices/names for group plots. Default is first 2 factors.
    colorset : list of str, optional
        Color palette for plots.
    legend_loc : str, default "best"
        Legend location.
    figsize : tuple, default (10, 6)
        Figure size in inches.

    Examples
    --------
    >>> from facmodts import fit_tsfm, plot_tsfm
    >>> # ... fit model ...
    >>>
    >>> # Individual asset plots
    >>> plot_tsfm(model, which=1, plot_single=True, asset_name="Asset1")
    >>>
    >>> # Group plots
    >>> plot_tsfm(model, which=[1, 2, 4])
    """
    # Default colorset
    if colorset is None:
        colorset = [
            "royalblue", "dimgray", "olivedrab", "firebrick",
            "goldenrod", "mediumorchid", "deepskyblue", "chocolate"
        ]

    # Ensure which is a list
    if which is None:
        which = [1]
    elif isinstance(which, int):
        which = [which]

    if plot_single:
        _plot_single_asset(model, which, asset_name, colorset, legend_loc, figsize)
    else:
        _plot_group_assets(model, which, a_sub, f_sub, colorset, legend_loc, figsize)


def _plot_single_asset(
    model: TsfmModel,
    which: List[int],
    asset_name: Optional[str],
    colorset: List[str],
    legend_loc: str,
    figsize: tuple,
) -> None:
    """Plot individual asset diagnostics."""
    # Determine asset
    if asset_name is None:
        if len(model.asset_names) > 1:
            raise ValueError(
                "asset_name is required when plot_single=True and model has multiple assets"
            )
        asset_name = model.asset_names[0]

    if asset_name not in model.asset_names:
        raise ValueError(f"Asset '{asset_name}' not found in model")

    # Get asset index
    i = model.asset_names.index(asset_name)

    # Get data for this asset
    asset_col = model.data[asset_name]
    valid_idx = ~asset_col.is_null()

    # Get fitted values and residuals
    factor_data = model.data.select(model.factor_names).to_numpy()
    alpha = model.alpha[i]
    beta = model.beta[i, :]
    beta_clean = np.where(np.isnan(beta), 0, beta)

    fitted = alpha + factor_data @ beta_clean
    actual = asset_col.to_numpy()
    residuals = actual - fitted

    # Remove NaN
    valid_mask = ~np.isnan(residuals)
    residuals_clean = residuals[valid_mask]
    fitted_clean = fitted[valid_mask]
    actual_clean = actual[valid_mask]

    resid_sd = model.resid_sd[i]

    # Create plots based on which
    for plot_num in which:
        fig = plt.figure(figsize=figsize)

        if plot_num == 1:
            # Actual and fitted time series
            plt.plot(actual_clean, label="Actual", color=colorset[0], linewidth=1.5)
            plt.plot(fitted_clean, label="Fitted", color=colorset[1], linewidth=1.5, linestyle="--")
            plt.xlabel("Time")
            plt.ylabel("Returns")
            plt.title(f"Actual and Fitted: {asset_name}")
            plt.legend(loc=legend_loc)
            plt.grid(True, alpha=0.3)

        elif plot_num == 2:
            # Actual vs fitted scatter
            plt.scatter(fitted_clean, actual_clean, color=colorset[0], alpha=0.6)
            plt.plot([fitted_clean.min(), fitted_clean.max()],
                    [fitted_clean.min(), fitted_clean.max()],
                    'k--', linewidth=1, label="45° line")
            plt.xlabel("Fitted")
            plt.ylabel("Actual")
            plt.title(f"Actual vs Fitted: {asset_name}")
            plt.legend(loc=legend_loc)
            plt.grid(True, alpha=0.3)

        elif plot_num == 3:
            # Residuals vs fitted
            plt.scatter(fitted_clean, residuals_clean, color=colorset[0], alpha=0.6)
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

            # Add lowess smoother
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(residuals_clean, fitted_clean, frac=0.3)
                plt.plot(smoothed[:, 0], smoothed[:, 1], color=colorset[1], linewidth=2)
            except ImportError:
                pass

            plt.xlabel("Fitted")
            plt.ylabel("Residuals")
            plt.title(f"Residuals vs Fitted: {asset_name}")
            plt.grid(True, alpha=0.3)

        elif plot_num == 4:
            # Residuals with SE bands
            plt.plot(residuals_clean, color=colorset[0], linewidth=1.5)
            plt.axhline(y=1.96*resid_sd, color=colorset[1], linestyle=':', linewidth=2, label="±1.96σ")
            plt.axhline(y=-1.96*resid_sd, color=colorset[1], linestyle=':', linewidth=2)
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
            plt.xlabel("Time")
            plt.ylabel("Residuals")
            plt.title(f"Residuals: {asset_name}")
            plt.legend(loc=legend_loc)
            plt.grid(True, alpha=0.3)

        elif plot_num == 5:
            # Squared residuals time series
            plt.plot(residuals_clean**2, color=colorset[0], linewidth=1.5)
            plt.xlabel("Time")
            plt.ylabel("Squared Residuals")
            plt.title(f"Squared Residuals: {asset_name}")
            plt.grid(True, alpha=0.3)

        elif plot_num == 6:
            # Absolute residuals time series
            plt.plot(np.abs(residuals_clean), color=colorset[0], linewidth=1.5)
            plt.xlabel("Time")
            plt.ylabel("Absolute Residuals")
            plt.title(f"Absolute Residuals: {asset_name}")
            plt.grid(True, alpha=0.3)

        elif plot_num == 7:
            # ACF/PACF of residuals
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            plot_acf(residuals_clean, lags=15, ax=ax1, color=colorset[0])
            ax1.set_title(f"ACF of Residuals: {asset_name}")

            plot_pacf(residuals_clean, lags=15, ax=ax2, color=colorset[0])
            ax2.set_title(f"PACF of Residuals: {asset_name}")

            plt.tight_layout()

        elif plot_num == 8:
            # ACF/PACF of squared residuals
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            plot_acf(residuals_clean**2, lags=15, ax=ax1, color=colorset[0])
            ax1.set_title(f"ACF of Squared Residuals: {asset_name}")

            plot_pacf(residuals_clean**2, lags=15, ax=ax2, color=colorset[0])
            ax2.set_title(f"PACF of Squared Residuals: {asset_name}")

            plt.tight_layout()

        elif plot_num == 9:
            # Histogram of residuals
            plt.hist(residuals_clean, bins=30, density=True, alpha=0.6,
                    color=colorset[0], edgecolor='black', label="Residuals")

            # Overlay normal distribution
            x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
            plt.plot(x, stats.norm.pdf(x, loc=residuals_clean.mean(), scale=resid_sd),
                    color=colorset[1], linewidth=2, linestyle='--', label="Normal")

            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(residuals_clean)
            plt.plot(x, kde(x), color=colorset[2], linewidth=2, label="KDE")

            plt.xlabel("Residuals")
            plt.ylabel("Density")
            plt.title(f"Histogram of Residuals: {asset_name}")
            plt.legend(loc=legend_loc)
            plt.grid(True, alpha=0.3)

        elif plot_num == 10:
            # QQ-plot
            stats.probplot(residuals_clean, dist="norm", plot=plt)
            plt.title(f"QQ-plot of Residuals: {asset_name}")
            plt.grid(True, alpha=0.3)

        else:
            warnings.warn(f"Plot option {plot_num} not recognized for individual plots")
            plt.close(fig)
            continue

        plt.tight_layout()
        plt.show()


def _plot_group_assets(
    model: TsfmModel,
    which: List[int],
    a_sub: Optional[Union[List[int], List[str]]],
    f_sub: Optional[Union[List[int], List[str]]],
    colorset: List[str],
    legend_loc: str,
    figsize: tuple,
) -> None:
    """Plot group asset diagnostics."""
    n_assets = len(model.asset_names)
    n_factors = len(model.factor_names)

    # Default subsets
    if a_sub is None:
        a_sub = list(range(min(6, n_assets)))
    if f_sub is None:
        f_sub = list(range(min(2, n_factors)))

    # Convert to indices
    if all(isinstance(x, str) for x in a_sub):
        a_sub = [model.asset_names.index(a) for a in a_sub]
    if all(isinstance(x, str) for x in f_sub):
        f_sub = [model.factor_names.index(f) for f in f_sub]

    # Validate
    if len(a_sub) < 2:
        raise ValueError("At least 2 assets required for group plots")

    asset_names_sub = [model.asset_names[i] for i in a_sub]
    factor_names_sub = [model.factor_names[i] for i in f_sub]

    # Create plots
    for plot_num in which:
        fig = plt.figure(figsize=figsize)

        if plot_num == 1:
            # Alpha coefficients
            alphas = model.alpha[a_sub]
            plt.barh(asset_names_sub, alphas, color=colorset[0])
            plt.xlabel("Alpha")
            plt.title("Factor Model Alpha (Intercept)")
            plt.grid(True, alpha=0.3, axis='x')

        elif plot_num == 2:
            # Beta coefficients
            n_factors_sub = len(f_sub)
            n_assets_sub = len(a_sub)

            if n_factors_sub == 1:
                # Single factor - simple barplot
                betas = model.beta[a_sub, f_sub[0]]
                plt.barh(asset_names_sub, betas, color=colorset[0])
                plt.xlabel(f"Beta ({factor_names_sub[0]})")
                plt.title("Factor Model Betas")
                plt.grid(True, alpha=0.3, axis='x')
            else:
                # Multiple factors - grouped barplot
                x = np.arange(len(asset_names_sub))
                width = 0.8 / n_factors_sub

                for j, f_idx in enumerate(f_sub):
                    betas = model.beta[a_sub, f_idx]
                    offset = (j - n_factors_sub/2 + 0.5) * width
                    plt.bar(x + offset, betas, width, label=factor_names_sub[j],
                           color=colorset[j % len(colorset)])

                plt.xlabel("Assets")
                plt.ylabel("Beta")
                plt.title("Factor Model Betas")
                plt.xticks(x, asset_names_sub, rotation=45, ha='right')
                plt.legend(loc=legend_loc)
                plt.grid(True, alpha=0.3, axis='y')

        elif plot_num == 3:
            # Actual and fitted (multiple assets)
            n_rows = (len(a_sub) + 1) // 2
            n_cols = 2 if len(a_sub) > 1 else 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows/2))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for idx, i in enumerate(a_sub):
                asset_name = model.asset_names[i]

                # Compute fitted
                factor_data = model.data.select(model.factor_names).to_numpy()
                alpha = model.alpha[i]
                beta = model.beta[i, :]
                beta_clean = np.where(np.isnan(beta), 0, beta)
                fitted = alpha + factor_data @ beta_clean
                actual = model.data[asset_name].to_numpy()

                axes[idx].plot(actual, label="Actual", color=colorset[0], linewidth=1.5)
                axes[idx].plot(fitted, label="Fitted", color=colorset[1],
                             linewidth=1.5, linestyle="--")
                axes[idx].set_title(f"{asset_name}")
                axes[idx].legend(loc=legend_loc, fontsize=8)
                axes[idx].grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(len(a_sub), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle("Actual and Fitted Returns")
            plt.tight_layout()

        elif plot_num == 4:
            # R-squared
            r2 = model.r_squared[a_sub]
            plt.barh(asset_names_sub, r2, color=colorset[0])
            plt.xlabel("R-squared")
            plt.xlim([0, 1])
            plt.title("R-squared Values")
            plt.grid(True, alpha=0.3, axis='x')

        elif plot_num == 5:
            # Residual volatility
            resid_sd = model.resid_sd[a_sub]
            plt.barh(asset_names_sub, resid_sd, color=colorset[0])
            plt.xlabel("Residual SD")
            plt.title("Residual Volatility")
            plt.grid(True, alpha=0.3, axis='x')

        elif plot_num == 6:
            # Residual correlation
            # Compute residuals for all assets
            residuals_matrix = []
            for i in a_sub:
                asset_name = model.asset_names[i]
                factor_data = model.data.select(model.factor_names).to_numpy()
                alpha = model.alpha[i]
                beta = model.beta[i, :]
                beta_clean = np.where(np.isnan(beta), 0, beta)
                fitted = alpha + factor_data @ beta_clean
                actual = model.data[asset_name].to_numpy()
                residuals = actual - fitted
                residuals_matrix.append(residuals)

            residuals_matrix = np.array(residuals_matrix).T

            # Remove NaN
            valid_mask = ~np.isnan(residuals_matrix).any(axis=1)
            residuals_clean = residuals_matrix[valid_mask, :]

            # Compute correlation
            corr = np.corrcoef(residuals_clean.T)

            # Plot heatmap
            im = plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im)
            plt.xticks(range(len(asset_names_sub)), asset_names_sub, rotation=45, ha='right')
            plt.yticks(range(len(asset_names_sub)), asset_names_sub)
            plt.title("Residual Correlation")

            # Add correlation values
            for i in range(len(asset_names_sub)):
                for j in range(len(asset_names_sub)):
                    plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                            color="white" if abs(corr[i, j]) > 0.5 else "black")

        elif plot_num == 7:
            # Factor contribution to SD
            decomp = fm_sd_decomp(model)
            pc_sd = decomp["pc_sd"][a_sub, :]

            # Create stacked barplot
            bottom = np.zeros(len(a_sub))
            for j in range(pc_sd.shape[1]):
                label = model.factor_names[j] if j < len(model.factor_names) else "Specific"
                plt.barh(asset_names_sub, pc_sd[:, j], left=bottom,
                        label=label, color=colorset[j % len(colorset)])
                bottom += pc_sd[:, j]

            plt.xlabel("Percentage Contribution")
            plt.xlim([0, 100])
            plt.title("Factor % Contribution to SD")
            plt.legend(loc=legend_loc, fontsize=8)
            plt.grid(True, alpha=0.3, axis='x')

        elif plot_num == 8:
            # Factor contribution to VaR
            try:
                decomp = fm_var_decomp(model, p=0.05)
                pc_var = decomp["pc_var"][a_sub, :]

                bottom = np.zeros(len(a_sub))
                for j in range(pc_var.shape[1]):
                    label = model.factor_names[j] if j < len(model.factor_names) else "Specific"
                    plt.barh(asset_names_sub, pc_var[:, j], left=bottom,
                            label=label, color=colorset[j % len(colorset)])
                    bottom += pc_var[:, j]

                plt.xlabel("Percentage Contribution")
                plt.xlim([0, 100])
                plt.title("Factor % Contribution to VaR (p=0.05)")
                plt.legend(loc=legend_loc, fontsize=8)
                plt.grid(True, alpha=0.3, axis='x')
            except Exception as e:
                warnings.warn(f"Could not compute VaR decomposition: {e}")
                plt.close(fig)
                continue

        elif plot_num == 9:
            # Factor contribution to ES
            try:
                decomp = fm_es_decomp(model, p=0.05)
                pc_es = decomp["pc_es"][a_sub, :]

                bottom = np.zeros(len(a_sub))
                for j in range(pc_es.shape[1]):
                    label = model.factor_names[j] if j < len(model.factor_names) else "Specific"
                    plt.barh(asset_names_sub, pc_es[:, j], left=bottom,
                            label=label, color=colorset[j % len(colorset)])
                    bottom += pc_es[:, j]

                plt.xlabel("Percentage Contribution")
                plt.xlim([0, 100])
                plt.title("Factor % Contribution to ES (p=0.05)")
                plt.legend(loc=legend_loc, fontsize=8)
                plt.grid(True, alpha=0.3, axis='x')
            except Exception as e:
                warnings.warn(f"Could not compute ES decomposition: {e}")
                plt.close(fig)
                continue

        else:
            warnings.warn(f"Plot option {plot_num} not recognized for group plots")
            plt.close(fig)
            continue

        plt.tight_layout()
        plt.show()


def plot_pafm(
    pa: PaFm,
    which: int = 1,
    asset_name: Optional[str] = None,
    plot_single: bool = False,
    max_show: int = 6,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot performance attribution results.

    Parameters
    ----------
    pa : PaFm
        Performance attribution object from pa_fm().
    which : int, default 1
        Plot selection:
        1 = Cumulative attributed returns (barplot)
        2 = Time series of attributed returns
        3 = Waterfall chart of cumulative returns
    asset_name : str, optional
        Asset name for single asset plots.
    plot_single : bool, default False
        If True, plot single asset. If False, plot all assets.
    max_show : int, default 6
        Maximum number of assets to show in group plots.
    figsize : tuple, default (10, 6)
        Figure size in inches.

    Examples
    --------
    >>> from facmodts import fit_tsfm, pa_fm, plot_pafm
    >>> # ... fit model and compute attribution ...
    >>> plot_pafm(pa, which=1)
    """
    if plot_single:
        # Single asset plot
        if asset_name is None:
            if len(pa.asset_names) > 1:
                raise ValueError("asset_name required for plot_single=True with multiple assets")
            asset_name = pa.asset_names[0]

        if asset_name not in pa.asset_names:
            raise ValueError(f"Asset '{asset_name}' not found")

        i = pa.asset_names.index(asset_name)

        fig = plt.figure(figsize=figsize)

        if which == 1:
            # Cumulative attributed returns
            values = np.concatenate([[pa.cum_spec_ret[i]], pa.cum_ret_attr_f[i, :]])
            labels = ["Specific"] + pa.factor_names
            colors = plt.cm.Set3(range(len(labels)))

            plt.barh(labels, values, color=colors)
            plt.xlabel("Cumulative Return")
            plt.title(f"Cumulative Attributed Returns: {asset_name}")
            plt.grid(True, alpha=0.3, axis='x')

        elif which == 2:
            # Time series of attributed returns
            attr_ts = pa.attr_list[asset_name]

            for factor in pa.factor_names:
                plt.plot(attr_ts[factor].to_numpy(), label=factor, linewidth=1.5)
            plt.plot(attr_ts["specific_returns"].to_numpy(),
                    label="Specific", linewidth=1.5, linestyle='--')

            plt.xlabel("Time")
            plt.ylabel("Attributed Returns")
            plt.title(f"Time Series of Attributed Returns: {asset_name}")
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)

        elif which == 3:
            # Waterfall chart
            values = np.concatenate([[pa.cum_spec_ret[i]], pa.cum_ret_attr_f[i, :]])
            labels = ["Specific"] + pa.factor_names

            cumsum = np.cumsum(values)
            starts = np.concatenate([[0], cumsum[:-1]])

            colors = ['red' if v < 0 else 'green' for v in values]

            for i, (start, val, label) in enumerate(zip(starts, values, labels)):
                plt.bar(i, val, bottom=start, color=colors[i], edgecolor='black')
                plt.text(i, start + val/2, f"{val:.2%}", ha='center', va='center')

            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.ylabel("Cumulative Return")
            plt.title(f"Waterfall Chart: {asset_name}")
            plt.axhline(y=0, color='black', linewidth=0.8)
            plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    else:
        # Group plots
        n_assets = min(len(pa.asset_names), max_show)
        asset_subset = pa.asset_names[:n_assets]

        if which == 1:
            # Cumulative attributed returns for multiple assets
            n_rows = (n_assets + 1) // 2
            n_cols = 2 if n_assets > 1 else 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows/2))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, asset_name in enumerate(asset_subset):
                i = pa.asset_names.index(asset_name)
                values = np.concatenate([[pa.cum_spec_ret[i]], pa.cum_ret_attr_f[i, :]])
                labels = ["Specific"] + pa.factor_names
                colors = plt.cm.Set3(range(len(labels)))

                axes[idx].barh(labels, values, color=colors)
                axes[idx].set_xlabel("Cumulative Return", fontsize=8)
                axes[idx].set_title(asset_name, fontsize=9)
                axes[idx].grid(True, alpha=0.3, axis='x')
                axes[idx].tick_params(labelsize=8)

            # Hide unused subplots
            for idx in range(n_assets, len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle("Cumulative Attributed Returns")
            plt.tight_layout()
            plt.show()

        elif which == 2:
            # Time series for multiple assets
            n_rows = (n_assets + 1) // 2
            n_cols = 2 if n_assets > 1 else 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows/2))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, asset_name in enumerate(asset_subset):
                attr_ts = pa.attr_list[asset_name]

                for factor in pa.factor_names:
                    axes[idx].plot(attr_ts[factor].to_numpy(), label=factor, linewidth=1)
                axes[idx].plot(attr_ts["specific_returns"].to_numpy(),
                             label="Specific", linewidth=1, linestyle='--')

                axes[idx].set_title(asset_name, fontsize=9)
                axes[idx].legend(loc="best", fontsize=7)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].tick_params(labelsize=8)

            # Hide unused subplots
            for idx in range(n_assets, len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle("Time Series of Attributed Returns")
            plt.tight_layout()
            plt.show()


def plot_tsfm_updn(
    updn_model: TsfmUpDn,
    asset_name: Optional[str] = None,
    add_sfm: bool = False,
    legend_loc: str = "best",
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot up/down market factor model results.

    Parameters
    ----------
    updn_model : TsfmUpDn
        Up/down market model from fit_tsfm_up_dn().
    asset_name : str, optional
        Asset to plot. If None, plots first asset.
    add_sfm : bool, default False
        If True, add single factor model line for comparison.
    legend_loc : str, default "best"
        Legend location.
    figsize : tuple, default (10, 6)
        Figure size in inches.

    Examples
    --------
    >>> from facmodts import fit_tsfm_up_dn, plot_tsfm_updn
    >>> # ... fit up/down model ...
    >>> plot_tsfm_updn(updn_model, asset_name="Asset1")
    """
    from .fitting import fit_tsfm

    # Determine asset
    if asset_name is None:
        asset_name = updn_model.up_model.asset_names[0]

    # Get market name
    mkt_name = updn_model.market_name

    # Get up/down fitted values
    up_data = updn_model.up_model.data
    dn_data = updn_model.down_model.data

    # Compute fitted for up market
    i_up = updn_model.up_model.asset_names.index(asset_name)
    factor_data_up = up_data.select(updn_model.up_model.factor_names).to_numpy()
    alpha_up = updn_model.up_model.alpha[i_up]
    beta_up = updn_model.up_model.beta[i_up, :]
    fitted_up = alpha_up + factor_data_up @ beta_up
    actual_up = up_data[asset_name].to_numpy()
    mkt_up = up_data[mkt_name].to_numpy()

    # Compute fitted for down market
    i_dn = updn_model.down_model.asset_names.index(asset_name)
    factor_data_dn = dn_data.select(updn_model.down_model.factor_names).to_numpy()
    alpha_dn = updn_model.down_model.alpha[i_dn]
    beta_dn = updn_model.down_model.beta[i_dn, :]
    fitted_dn = alpha_dn + factor_data_dn @ beta_dn
    actual_dn = dn_data[asset_name].to_numpy()
    mkt_dn = dn_data[mkt_name].to_numpy()

    # Create plot
    fig = plt.figure(figsize=figsize)

    # Plot actual values
    plt.scatter(mkt_up, actual_up, color='blue', alpha=0.6, label='Up Market (Actual)')
    plt.scatter(mkt_dn, actual_dn, color='red', alpha=0.6, label='Down Market (Actual)')

    # Plot fitted lines
    plt.plot(mkt_up, fitted_up, color='blue', linewidth=2, linestyle='--', label='Up Market (Fitted)')
    plt.plot(mkt_dn, fitted_dn, color='red', linewidth=2, linestyle='--', label='Down Market (Fitted)')

    # Add reference lines
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    plt.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    # Add single factor model if requested
    if add_sfm:
        # Combine data
        full_data = pl.concat([up_data, dn_data])
        sfm = fit_tsfm(
            asset_names=[asset_name],
            factor_names=[mkt_name],
            data=full_data,
            fit_method=updn_model.up_model.fit_method,
        )

        # Get fitted
        i_sfm = 0
        factor_data_sfm = full_data.select([mkt_name]).to_numpy()
        alpha_sfm = sfm.alpha[i_sfm]
        beta_sfm = sfm.beta[i_sfm, 0]

        # Plot line
        x_range = np.linspace(min(mkt_up.min(), mkt_dn.min()),
                             max(mkt_up.max(), mkt_dn.max()), 100)
        y_range = alpha_sfm + beta_sfm * x_range
        plt.plot(x_range, y_range, color='green', linewidth=2,
                linestyle=':', label='Single Factor Model')

    plt.xlabel(f"{mkt_name} Returns")
    plt.ylabel(f"{asset_name} Returns")
    plt.title(f"Up/Down Market Model: {asset_name}")
    plt.legend(loc=legend_loc)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
