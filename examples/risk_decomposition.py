"""
Risk decomposition examples for facmodts package.

This script demonstrates factor model covariance and risk decomposition
for SD, VaR, and ES using Euler's theorem.
"""

import numpy as np
import polars as pl
from datetime import date, timedelta

from facmodts import (
    fit_tsfm,
    fm_cov,
    fm_sd_decomp,
    fm_var_decomp,
    fm_es_decomp
)


def create_portfolio_returns(n_periods=250, n_assets=3, n_factors=2):
    """
    Create synthetic portfolio return data.

    Parameters:
        n_periods: Number of time periods
        n_assets: Number of assets
        n_factors: Number of factors

    Returns:
        Polars DataFrame with asset and factor returns.
    """
    # Generate dates
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_periods)]

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate factor returns (macro factors)
    factor1 = 0.005 + 0.02 * np.sin(np.linspace(0, 8 * np.pi, n_periods)) + 0.015 * np.random.randn(n_periods)
    factor2 = -0.002 + 0.015 * np.cos(np.linspace(0, 6 * np.pi, n_periods)) + 0.012 * np.random.randn(n_periods)

    # Generate asset returns with known factor loadings
    # Asset1: high beta to Factor1 (growth stock)
    # Asset2: balanced exposure
    # Asset3: defensive (low betas)
    asset1 = 0.002 + 1.2 * factor1 + 0.3 * factor2 + 0.012 * np.random.randn(n_periods)
    asset2 = 0.001 + 0.8 * factor1 + 0.6 * factor2 + 0.010 * np.random.randn(n_periods)
    asset3 = 0.001 + 0.4 * factor1 - 0.2 * factor2 + 0.008 * np.random.randn(n_periods)

    data = pl.DataFrame({
        "date": dates,
        "Growth": asset1,
        "Balanced": asset2,
        "Defensive": asset3,
        "Market": factor1,
        "Value": factor2,
    })

    return data


def example1_covariance_matrix():
    """Example 1: Factor model covariance matrix."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Factor Model Covariance Matrix")
    print("="*80)

    data = create_portfolio_returns(n_periods=250)

    # Fit factor model
    model = fit_tsfm(
        data=data,
        asset_names=["Growth", "Balanced", "Defensive"],
        factor_names=["Market", "Value"],
        fit_method="LS"
    )

    # Compute covariance matrix
    cov_matrix = fm_cov(model)

    print("\nFactor model covariance matrix (3 assets):")
    print("-" * 60)
    print(f"{'':>12} {'Growth':>12} {'Balanced':>12} {'Defensive':>12}")
    print("-" * 60)

    asset_names = ["Growth", "Balanced", "Defensive"]
    for i, asset in enumerate(asset_names):
        row_str = f"{asset:>12}"
        for j in range(3):
            row_str += f" {cov_matrix[i, j]:>12.6f}"
        print(row_str)

    print("\nStandard deviations (sqrt of diagonal):")
    for i, asset in enumerate(asset_names):
        sd = np.sqrt(cov_matrix[i, i])
        print(f"  {asset}: {sd:.4f} ({sd*100:.2f}% annualized)")

    print("\nCorrelation matrix:")
    print("-" * 60)
    # Compute correlation from covariance
    D_inv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr_matrix = D_inv @ cov_matrix @ D_inv

    print(f"{'':>12} {'Growth':>12} {'Balanced':>12} {'Defensive':>12}")
    print("-" * 60)
    for i, asset in enumerate(asset_names):
        row_str = f"{asset:>12}"
        for j in range(3):
            row_str += f" {corr_matrix[i, j]:>12.4f}"
        print(row_str)


def example2_sd_decomposition():
    """Example 2: Standard deviation decomposition."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Standard Deviation Decomposition")
    print("="*80)

    data = create_portfolio_returns(n_periods=250)

    model = fit_tsfm(
        data=data,
        asset_names=["Growth", "Balanced", "Defensive"],
        factor_names=["Market", "Value"],
        fit_method="LS"
    )

    # SD decomposition
    sd_decomp = fm_sd_decomp(model)

    print("\nSD decomposition by factor (Euler's theorem):")
    print("-" * 80)
    print(f"{'Asset':<12} {'SD':<10} {'Market %':<12} {'Value %':<12} {'Residual %':<12}")
    print("-" * 80)

    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        sd = sd_decomp["sd_fm"][i]
        pc_market = sd_decomp["pc_sd"][i, 0]
        pc_value = sd_decomp["pc_sd"][i, 1]
        pc_resid = sd_decomp["pc_sd"][i, 2]

        print(f"{asset:<12} {sd:<10.4f} {pc_market:<12.2f} {pc_value:<12.2f} {pc_resid:<12.2f}")

    print("\nComponent SD (cSD = mSD * beta):")
    print("-" * 80)
    print(f"{'Asset':<12} {'Market':<12} {'Value':<12} {'Residual':<12} {'Sum':<12}")
    print("-" * 80)

    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        c_market = sd_decomp["c_sd"][i, 0]
        c_value = sd_decomp["c_sd"][i, 1]
        c_resid = sd_decomp["c_sd"][i, 2]
        total = c_market + c_value + c_resid

        print(f"{asset:<12} {c_market:<12.4f} {c_value:<12.4f} {c_resid:<12.4f} {total:<12.4f}")

    print("\nInterpretation:")
    print("  - Growth: Dominated by Market factor (high beta)")
    print("  - Balanced: Mixed exposure to both factors")
    print("  - Defensive: Lower overall SD, less Market sensitivity")


def example3_var_decomposition():
    """Example 3: Value-at-Risk decomposition."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Value-at-Risk (VaR) Decomposition")
    print("="*80)

    data = create_portfolio_returns(n_periods=250)

    model = fit_tsfm(
        data=data,
        asset_names=["Growth", "Balanced", "Defensive"],
        factor_names=["Market", "Value"],
        fit_method="LS"
    )

    # VaR decomposition (5% tail)
    var_decomp = fm_var_decomp(model, p=0.05, type="np")

    print("\n5% VaR decomposition (non-parametric):")
    print("-" * 80)
    print(f"{'Asset':<12} {'VaR (5%)':<12} {'Market %':<12} {'Value %':<12} {'Residual %':<12}")
    print("-" * 80)

    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        var = var_decomp["var_fm"][i]
        pc_market = var_decomp["pc_var"][i, 0]
        pc_value = var_decomp["pc_var"][i, 1]
        pc_resid = var_decomp["pc_var"][i, 2]

        print(f"{asset:<12} {var:<12.4f} {pc_market:<12.2f} {pc_value:<12.2f} {pc_resid:<12.2f}")

    print("\nComponent VaR (cVaR = mVaR * beta):")
    print("-" * 80)
    print(f"{'Asset':<12} {'Market':<12} {'Value':<12} {'Residual':<12} {'Sum':<12}")
    print("-" * 80)

    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        c_market = var_decomp["c_var"][i, 0]
        c_value = var_decomp["c_var"][i, 1]
        c_resid = var_decomp["c_var"][i, 2]
        total = c_market + c_value + c_resid

        print(f"{asset:<12} {c_market:<12.4f} {c_value:<12.4f} {c_resid:<12.4f} {total:<12.4f}")

    print(f"\nNumber of VaR exceedances (expected ~12-13 out of 250):")
    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        n_exceed = var_decomp["n_exceed"][i]
        print(f"  {asset}: {n_exceed} exceedances ({n_exceed/250*100:.1f}%)")


def example4_es_decomposition():
    """Example 4: Expected Shortfall decomposition."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Expected Shortfall (ES/CVaR) Decomposition")
    print("="*80)

    data = create_portfolio_returns(n_periods=250)

    model = fit_tsfm(
        data=data,
        asset_names=["Growth", "Balanced", "Defensive"],
        factor_names=["Market", "Value"],
        fit_method="LS"
    )

    # ES decomposition (5% tail)
    es_decomp = fm_es_decomp(model, p=0.05, type="np")

    print("\n5% Expected Shortfall decomposition (non-parametric):")
    print("-" * 80)
    print(f"{'Asset':<12} {'ES (5%)':<12} {'Market %':<12} {'Value %':<12} {'Residual %':<12}")
    print("-" * 80)

    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        es = es_decomp["es_fm"][i]
        pc_market = es_decomp["pc_es"][i, 0]
        pc_value = es_decomp["pc_es"][i, 1]
        pc_resid = es_decomp["pc_es"][i, 2]

        print(f"{asset:<12} {es:<12.4f} {pc_market:<12.2f} {pc_value:<12.2f} {pc_resid:<12.2f}")

    print("\nComponent ES (cES = mES * beta):")
    print("-" * 80)
    print(f"{'Asset':<12} {'Market':<12} {'Value':<12} {'Residual':<12} {'Sum':<12}")
    print("-" * 80)

    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        c_market = es_decomp["c_es"][i, 0]
        c_value = es_decomp["c_es"][i, 1]
        c_resid = es_decomp["c_es"][i, 2]
        total = c_market + c_value + c_resid

        print(f"{asset:<12} {c_market:<12.4f} {c_value:<12.4f} {c_resid:<12.4f} {total:<12.4f}")

    print("\nInterpretation:")
    print("  - ES (CVaR) measures expected loss in worst 5% of cases")
    print("  - ES >= VaR (more conservative risk measure)")
    print("  - Factor contributions show tail risk drivers")


def example5_np_vs_normal():
    """Example 5: Compare non-parametric vs normal VaR/ES."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Non-Parametric vs Normal VaR/ES")
    print("="*80)

    data = create_portfolio_returns(n_periods=250)

    model = fit_tsfm(
        data=data,
        asset_names=["Growth"],
        factor_names=["Market", "Value"],
        fit_method="LS"
    )

    # Non-parametric
    var_np = fm_var_decomp(model, p=0.05, type="np")
    es_np = fm_es_decomp(model, p=0.05, type="np")

    # Normal (parametric)
    var_normal = fm_var_decomp(model, p=0.05, type="normal")
    es_normal = fm_es_decomp(model, p=0.05, type="normal")

    print("\nComparison for Growth asset:")
    print("-" * 60)
    print(f"{'Metric':<20} {'Non-Parametric':<20} {'Normal':<20}")
    print("-" * 60)
    print(f"{'5% VaR':<20} {var_np['var_fm'][0]:<20.4f} {var_normal['var_fm'][0]:<20.4f}")
    print(f"{'5% ES':<20} {es_np['es_fm'][0]:<20.4f} {es_normal['es_fm'][0]:<20.4f}")

    print("\nPercentage contributions (Non-Parametric VaR):")
    print(f"  Market: {var_np['pc_var'][0, 0]:.2f}%")
    print(f"  Value: {var_np['pc_var'][0, 1]:.2f}%")
    print(f"  Residual: {var_np['pc_var'][0, 2]:.2f}%")

    print("\nPercentage contributions (Normal VaR):")
    print(f"  Market: {var_normal['pc_var'][0, 0]:.2f}%")
    print(f"  Value: {var_normal['pc_var'][0, 1]:.2f}%")
    print(f"  Residual: {var_normal['pc_var'][0, 2]:.2f}%")

    print("\nNote: Normal assumes Gaussian distributions;")
    print("      Non-parametric captures actual distribution including fat tails.")


def example6_risk_hierarchy():
    """Example 6: Demonstrate SD <= VaR <= ES hierarchy."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Risk Measure Hierarchy (SD, VaR, ES)")
    print("="*80)

    data = create_portfolio_returns(n_periods=250)

    model = fit_tsfm(
        data=data,
        asset_names=["Growth", "Balanced", "Defensive"],
        factor_names=["Market", "Value"],
        fit_method="LS"
    )

    # All risk measures
    sd_decomp = fm_sd_decomp(model)
    var_decomp = fm_var_decomp(model, p=0.05, type="np")
    es_decomp = fm_es_decomp(model, p=0.05, type="np")

    print("\nRisk measures for all assets:")
    print("-" * 80)
    print(f"{'Asset':<12} {'SD':<15} {'5% VaR':<15} {'5% ES':<15} {'ES/SD Ratio':<15}")
    print("-" * 80)

    for i, asset in enumerate(["Growth", "Balanced", "Defensive"]):
        sd = sd_decomp["sd_fm"][i]
        var = abs(var_decomp["var_fm"][i])  # Absolute value for comparison
        es = abs(es_decomp["es_fm"][i])
        es_sd_ratio = es / sd

        print(f"{asset:<12} {sd:<15.4f} {var:<15.4f} {es:<15.4f} {es_sd_ratio:<15.2f}")

    print("\nObservations:")
    print("  1. ES >= VaR (ES is more conservative)")
    print("  2. ES/SD ratio ~2-3x for normal tails, higher for fat tails")
    print("  3. All satisfy Euler decomposition: Risk = sum(components)")


def example7_robust_risk_decomposition():
    """Example 7: Risk decomposition with robust regression."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Risk Decomposition with Robust Regression")
    print("="*80)

    # Create data with outliers
    data = create_portfolio_returns(n_periods=250)

    # Add outliers to Growth asset
    growth_values = data["Growth"].to_numpy().copy()
    outlier_idx = np.random.choice(250, size=10, replace=False)
    growth_values[outlier_idx] += np.random.choice([-0.08, 0.08], size=10)
    data = data.with_columns(pl.Series("Growth", growth_values))

    # Fit LS model
    model_ls = fit_tsfm(
        data=data,
        asset_names=["Growth"],
        factor_names=["Market", "Value"],
        fit_method="LS"
    )

    # Fit Robust model
    model_robust = fit_tsfm(
        data=data,
        asset_names=["Growth"],
        factor_names=["Market", "Value"],
        fit_method="Robust",
        family="bisquare"
    )

    # VaR decomposition for both
    var_ls = fm_var_decomp(model_ls, p=0.05, type="np")
    var_robust = fm_var_decomp(model_robust, p=0.05, type="np")

    print("\n5% VaR comparison (Growth asset with outliers):")
    print("-" * 60)
    print(f"{'Method':<15} {'VaR':<15} {'Market %':<12} {'Value %':<12} {'Residual %':<12}")
    print("-" * 60)

    print(f"{'LS':<15} {var_ls['var_fm'][0]:<15.4f} "
          f"{var_ls['pc_var'][0, 0]:<12.2f} "
          f"{var_ls['pc_var'][0, 1]:<12.2f} "
          f"{var_ls['pc_var'][0, 2]:<12.2f}")

    print(f"{'Robust':<15} {var_robust['var_fm'][0]:<15.4f} "
          f"{var_robust['pc_var'][0, 0]:<12.2f} "
          f"{var_robust['pc_var'][0, 1]:<12.2f} "
          f"{var_robust['pc_var'][0, 2]:<12.2f}")

    print("\nNote: Robust regression resists outliers, leading to different")
    print("      factor exposures and risk decompositions.")


if __name__ == "__main__":
    print("\nfacmodts Package: Risk Decomposition Examples")
    print("=" * 80)

    example1_covariance_matrix()
    example2_sd_decomposition()
    example3_var_decomposition()
    example4_es_decomposition()
    example5_np_vs_normal()
    example6_risk_hierarchy()
    example7_robust_risk_decomposition()

    print("\n" + "="*80)
    print("All risk decomposition examples completed!")
    print("="*80 + "\n")
