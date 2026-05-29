"""
Examples demonstrating visualization methods for factor models.

This script shows how to:
1. Create diagnostic plots for fitted models
2. Visualize performance attribution
3. Plot up/down market models
4. Generate risk decomposition visualizations
"""

import numpy as np
import polars as pl
from datetime import date, timedelta
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt

from facmodts import (
    fit_tsfm,
    pa_fm,
    fit_tsfm_up_dn,
    plot_tsfm,
    plot_pafm,
    plot_tsfm_updn,
)


def create_portfolio_data(n_periods=120, seed=42):
    """
    Create sample portfolio data with multiple funds.

    Returns
    -------
    polars.DataFrame
        Portfolio returns and factor data.
    """
    np.random.seed(seed)

    dates = [date(2020, 1, 1) + timedelta(days=i * 7) for i in range(n_periods)]

    # Fama-French style factors
    mkt = 0.002 + 0.03 * np.random.randn(n_periods)
    smb = 0.001 + 0.015 * np.random.randn(n_periods)
    hml = -0.0005 + 0.012 * np.random.randn(n_periods)

    # Three funds with different styles
    growth_fund = 0.0025 + 1.1 * mkt - 0.15 * smb - 0.4 * hml + 0.008 * np.random.randn(n_periods)
    value_fund = 0.0015 + 0.85 * mkt + 0.1 * smb + 0.5 * hml + 0.007 * np.random.randn(n_periods)
    balanced_fund = 0.002 + 0.95 * mkt + 0.05 * smb + 0.05 * hml + 0.006 * np.random.randn(n_periods)

    data = pl.DataFrame({
        "date": dates,
        "GrowthFund": growth_fund,
        "ValueFund": value_fund,
        "BalancedFund": balanced_fund,
        "Market": mkt,
        "Size": smb,
        "Value": hml,
    })

    return data


def example1_individual_asset_plots():
    """
    Example 1: Individual asset diagnostic plots.

    Demonstrates various diagnostic plots for a single asset.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Individual Asset Diagnostic Plots")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    # Fit model for one asset
    model = fit_tsfm(
        asset_names=["GrowthFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    print("\nGenerating diagnostic plots for GrowthFund...")

    # Plot 1: Actual and fitted time series
    print("\n1. Actual vs Fitted Time Series")
    plot_tsfm(model, which=1, plot_single=True, asset_name="GrowthFund")

    # Plot 2: Actual vs fitted scatter
    print("\n2. Actual vs Fitted Scatter")
    plot_tsfm(model, which=2, plot_single=True, asset_name="GrowthFund")

    # Plot 3: Residuals vs fitted
    print("\n3. Residuals vs Fitted")
    plot_tsfm(model, which=3, plot_single=True, asset_name="GrowthFund")

    # Plot 4: Residuals with SE bands
    print("\n4. Residuals with Standard Error Bands")
    plot_tsfm(model, which=4, plot_single=True, asset_name="GrowthFund")

    # Plot 9: Histogram of residuals
    print("\n5. Histogram of Residuals")
    plot_tsfm(model, which=9, plot_single=True, asset_name="GrowthFund")

    # Plot 10: QQ-plot
    print("\n6. QQ-plot of Residuals")
    plot_tsfm(model, which=10, plot_single=True, asset_name="GrowthFund")

    print("\nDiagnostic plots demonstrate:")
    print("- Model fit quality (actual vs fitted)")
    print("- Residual patterns (heteroskedasticity, outliers)")
    print("- Normality assumption (QQ-plot, histogram)")


def example2_group_plots():
    """
    Example 2: Group plots for multiple assets.

    Shows how to compare multiple assets using group plots.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Group Plots for Multiple Assets")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    # Fit model for all three funds
    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund", "BalancedFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    print("\nGenerating group comparison plots...")

    # Plot 1: Alpha coefficients
    print("\n1. Alpha Coefficients")
    plot_tsfm(model, which=1, plot_single=False)

    # Plot 2: Beta coefficients
    print("\n2. Beta Coefficients (Factor Exposures)")
    plot_tsfm(model, which=2, plot_single=False)

    # Plot 4: R-squared
    print("\n3. R-squared Comparison")
    plot_tsfm(model, which=4, plot_single=False)

    # Plot 5: Residual volatility
    print("\n4. Residual Volatility")
    plot_tsfm(model, which=5, plot_single=False)

    # Plot 6: Residual correlation
    print("\n5. Residual Correlation Matrix")
    plot_tsfm(model, which=6, plot_single=False)

    print("\nGroup plots show:")
    print("- GrowthFund: High market beta (1.1), negative value loading")
    print("- ValueFund: Lower market beta (0.85), positive value loading")
    print("- BalancedFund: Middle ground with balanced exposures")


def example3_risk_decomposition_plots():
    """
    Example 3: Risk decomposition visualizations.

    Shows factor contributions to SD, VaR, and ES.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Risk Decomposition Visualizations")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund", "BalancedFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    print("\nGenerating risk decomposition plots...")

    # Plot 7: Factor contribution to SD
    print("\n1. Factor Contribution to Standard Deviation")
    plot_tsfm(model, which=7, plot_single=False)

    print("\nStacked bar charts show percentage contribution of:")
    print("- Each factor to total portfolio risk")
    print("- Specific (idiosyncratic) risk")
    print("- Allows identification of dominant risk drivers")


def example4_performance_attribution():
    """
    Example 4: Performance attribution visualizations.

    Shows how to visualize factor contributions to returns.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Performance Attribution Visualizations")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund", "BalancedFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    # Compute attribution
    pa = pa_fm(model)

    print("\nGenerating performance attribution plots...")

    # Plot 1: Cumulative attributed returns
    print("\n1. Cumulative Attributed Returns")
    plot_pafm(pa, which=1, plot_single=False)

    # Plot 2: Time series of attributed returns
    print("\n2. Time Series of Attributed Returns")
    plot_pafm(pa, which=2, plot_single=False)

    # Single asset waterfall chart
    print("\n3. Waterfall Chart (GrowthFund)")
    plot_pafm(pa, which=3, plot_single=True, asset_name="GrowthFund")

    print("\nAttribution visualizations show:")
    print("- Contribution of each factor to cumulative returns")
    print("- Evolution of factor contributions over time")
    print("- Specific returns not explained by factors")


def example5_up_down_market():
    """
    Example 5: Up/down market model visualization.

    Shows asymmetric beta in up vs down markets.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Up/Down Market Asymmetry")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    # Fit up/down model
    updn_model = fit_tsfm_up_dn(
        asset_names=["GrowthFund", "ValueFund"],
        mkt_name="Market",
        data=data,
        fit_method="LS",
    )

    print("\nUp market periods:", sum(updn_model.up_market_flag))
    print("Down market periods:", sum(~updn_model.up_market_flag))

    # Get betas
    up_beta_growth = updn_model.up_model.beta[0, 0]
    dn_beta_growth = updn_model.down_model.beta[0, 0]

    print(f"\nGrowthFund asymmetry:")
    print(f"  Up market beta:   {up_beta_growth:.3f}")
    print(f"  Down market beta: {dn_beta_growth:.3f}")
    print(f"  Difference:       {up_beta_growth - dn_beta_growth:.3f}")

    # Plot up/down model
    print("\n1. Up/Down Market Plot")
    plot_tsfm_updn(updn_model, asset_name="GrowthFund")

    # Plot with single factor model comparison
    print("\n2. Up/Down with Single Factor Model")
    plot_tsfm_updn(updn_model, asset_name="GrowthFund", add_sfm=True)

    print("\nUp/down market plots reveal:")
    print("- Different sensitivities in bull vs bear markets")
    print("- Potential convexity in returns")
    print("- Market-timing opportunities")


def example6_acf_analysis():
    """
    Example 6: Autocorrelation analysis.

    Examines serial correlation in residuals.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Autocorrelation Analysis")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    model = fit_tsfm(
        asset_names=["GrowthFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    print("\nGenerating autocorrelation plots...")

    # Plot 7: ACF/PACF of residuals
    print("\n1. ACF/PACF of Residuals")
    plot_tsfm(model, which=7, plot_single=True, asset_name="GrowthFund")

    # Plot 8: ACF/PACF of squared residuals
    print("\n2. ACF/PACF of Squared Residuals")
    plot_tsfm(model, which=8, plot_single=True, asset_name="GrowthFund")

    print("\nACF analysis helps identify:")
    print("- Serial correlation in residuals (model misspecification)")
    print("- Volatility clustering (ARCH/GARCH effects)")
    print("- Need for dynamic models")


def example7_subset_selection():
    """
    Example 7: Plotting with asset and factor subsets.

    Shows how to focus on specific assets and factors.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Subset Selection in Plots")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund", "BalancedFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    print("\nPlotting subset of assets and factors...")

    # Plot only Growth and Value funds with Market and Size factors
    print("\n1. Betas for GrowthFund and ValueFund (Market and Size only)")
    plot_tsfm(model, which=2, plot_single=False,
             a_sub=["GrowthFund", "ValueFund"],
             f_sub=["Market", "Size"])

    # Plot only Growth and Balanced funds
    print("\n2. R-squared for GrowthFund and BalancedFund")
    plot_tsfm(model, which=4, plot_single=False,
             a_sub=["GrowthFund", "BalancedFund"])

    print("\nSubset selection allows:")
    print("- Focused comparison of specific assets")
    print("- Emphasis on key risk factors")
    print("- Cleaner visualizations for presentations")


if __name__ == "__main__":
    print("\nfacmodts Package: Visualization Examples")
    print("=" * 80)

    example1_individual_asset_plots()
    example2_group_plots()
    example3_risk_decomposition_plots()
    example4_performance_attribution()
    example5_up_down_market()
    example6_acf_analysis()
    example7_subset_selection()

    print("\n" + "=" * 80)
    print("All visualization examples completed!")
    print("=" * 80 + "\n")
