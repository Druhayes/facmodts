"""
Basic usage examples for facmodts package.

This script demonstrates the core functionality of the facmodts package,
including LS and DLS regression, excess returns computation, and result interpretation.
"""

import numpy as np
import polars as pl
from datetime import date, timedelta

from facmodts import fit_tsfm, fit_tsfm_control


def create_sample_data(n_periods=200, n_assets=3, n_factors=3):
    """
    Create synthetic return data for demonstration.

    Parameters:
        n_periods: Number of time periods (default 200)
        n_assets: Number of assets (default 3)
        n_factors: Number of factors (default 3)

    Returns:
        Polars DataFrame with dates, asset returns, factor returns, and RF.
    """
    # Generate dates
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_periods)]

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate factor returns (mean-reverting)
    factors = {}
    factor_names = [f"Factor{i+1}" for i in range(n_factors)]
    for fname in factor_names:
        factors[fname] = 0.001 + 0.02 * np.sin(np.linspace(0, 6 * np.pi, n_periods)) + 0.01 * np.random.randn(n_periods)

    # Risk-free rate (constant)
    rf = np.full(n_periods, 0.0002)  # 2 bps per day

    # Generate asset returns with known factor loadings
    assets = {}
    true_betas = {
        "Asset1": [0.8, 0.3, 0.0],
        "Asset2": [1.2, -0.5, 0.4],
        "Asset3": [0.5, 0.2, 0.8],
    }

    for asset_name, betas in true_betas.items():
        ret = 0.0005  # Alpha
        for i, fname in enumerate(factor_names):
            ret = ret + betas[i] * factors[fname]
        ret = ret + 0.01 * np.random.randn(n_periods)  # Idiosyncratic risk
        assets[asset_name] = ret

    # Create DataFrame
    data = pl.DataFrame({
        "date": dates,
        **assets,
        **factors,
        "RF": rf
    })

    return data, true_betas


def example1_basic_ls():
    """Example 1: Basic Ordinary Least Squares (LS) regression."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic LS Regression")
    print("="*80)

    # Create sample data
    data, true_betas = create_sample_data()

    # Fit LS model
    model = fit_tsfm(
        data=data,
        asset_names=["Asset1", "Asset2", "Asset3"],
        factor_names=["Factor1", "Factor2", "Factor3"],
        fit_method="LS"
    )

    # Display results
    print("\nEstimated Beta Coefficients:")
    print(f"{'Asset':<10} {'Factor1':<12} {'Factor2':<12} {'Factor3':<12}")
    print("-" * 50)
    for i, asset in enumerate(model.asset_names):
        print(f"{asset:<10} {model.beta[i, 0]:>11.4f} {model.beta[i, 1]:>11.4f} {model.beta[i, 2]:>11.4f}")

    print("\nTrue Beta Coefficients (for comparison):")
    for asset, betas in true_betas.items():
        print(f"{asset:<10} {betas[0]:>11.4f} {betas[1]:>11.4f} {betas[2]:>11.4f}")

    print("\nR-squared values:")
    for i, asset in enumerate(model.asset_names):
        print(f"  {asset}: {model.r_squared[i]:.4f}")

    print("\nResidual Standard Deviations:")
    for i, asset in enumerate(model.asset_names):
        print(f"  {asset}: {model.resid_sd[i]:.6f}")


def example2_dls():
    """Example 2: Discounted Least Squares (DLS) with exponential weights."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Discounted Least Squares (DLS)")
    print("="*80)

    data, _ = create_sample_data()

    # Fit DLS model with default decay (0.95)
    model_dls = fit_tsfm(
        data=data,
        asset_names=["Asset1"],
        factor_names=["Factor1", "Factor2"],
        fit_method="DLS"
    )

    # Fit DLS model with stronger decay (0.90 - more weight on recent data)
    model_dls_strong = fit_tsfm(
        data=data,
        asset_names=["Asset1"],
        factor_names=["Factor1", "Factor2"],
        fit_method="DLS",
        decay=0.90
    )

    # Compare with LS
    model_ls = fit_tsfm(
        data=data,
        asset_names=["Asset1"],
        factor_names=["Factor1", "Factor2"],
        fit_method="LS"
    )

    print("\nBeta Coefficients Comparison:")
    print(f"{'Method':<20} {'Factor1':<12} {'Factor2':<12} {'R²':<10}")
    print("-" * 60)
    print(f"{'LS':<20} {model_ls.beta[0, 0]:>11.4f} {model_ls.beta[0, 1]:>11.4f} {model_ls.r_squared[0]:>9.4f}")
    print(f"{'DLS (decay=0.95)':<20} {model_dls.beta[0, 0]:>11.4f} {model_dls.beta[0, 1]:>11.4f} {model_dls.r_squared[0]:>9.4f}")
    print(f"{'DLS (decay=0.90)':<20} {model_dls_strong.beta[0, 0]:>11.4f} {model_dls_strong.beta[0, 1]:>11.4f} {model_dls_strong.r_squared[0]:>9.4f}")

    print("\nNote: DLS gives more weight to recent observations,")
    print("      which can capture changing factor sensitivities.")


def example3_excess_returns():
    """Example 3: Fitting with excess returns (subtracting risk-free rate)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Excess Returns")
    print("="*80)

    data, _ = create_sample_data()

    # Fit with excess returns (rf_name specified)
    model_excess = fit_tsfm(
        data=data,
        asset_names=["Asset1", "Asset2"],
        factor_names=["Factor1", "Factor2", "Factor3"],
        rf_name="RF",
        fit_method="LS"
    )

    print("\nModel fitted using excess returns (R - RF)")
    print(f"\nBeta coefficients (excess return form):")
    print(f"{'Asset':<10} {'Factor1':<12} {'Factor2':<12} {'Factor3':<12} {'R²':<10}")
    print("-" * 60)
    for i, asset in enumerate(model_excess.asset_names):
        print(f"{asset:<10} {model_excess.beta[i, 0]:>11.4f} {model_excess.beta[i, 1]:>11.4f} {model_excess.beta[i, 2]:>11.4f} {model_excess.r_squared[i]:>9.4f}")

    print("\nNote: Excess returns are standard in factor models as they represent")
    print("      returns above the risk-free rate (i.e., the compensation for risk).")


def example4_control_parameters():
    """Example 4: Using control parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Control Parameters")
    print("="*80)

    data, _ = create_sample_data()

    # Create custom control parameters
    ctrl = fit_tsfm_control(
        decay=0.93,        # Custom decay for DLS
        bb=0.4,            # Breakdown point for robust (Phase 2)
        efficiency=0.90    # Efficiency for robust (Phase 2)
    )

    print("\nCustom control parameters:")
    print(f"  Decay factor (DLS): {ctrl.decay}")
    print(f"  Breakdown point (Robust): {ctrl.bb}")
    print(f"  Efficiency (Robust): {ctrl.efficiency}")

    # Fit model with custom control
    model = fit_tsfm(
        data=data,
        asset_names=["Asset1"],
        factor_names=["Factor1", "Factor2"],
        fit_method="DLS",
        control=ctrl
    )

    print(f"\nFitted model with custom decay={ctrl.decay}:")
    print(f"  Beta: {model.beta[0]}")
    print(f"  R²: {model.r_squared[0]:.4f}")


def example5_model_inspection():
    """Example 5: Inspecting fitted model objects."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Model Inspection")
    print("="*80)

    data, _ = create_sample_data()

    model = fit_tsfm(
        data=data,
        asset_names=["Asset1", "Asset2"],
        factor_names=["Factor1", "Factor2"],
        fit_method="LS"
    )

    print("\nModel attributes:")
    print(f"  Number of assets: {len(model.asset_names)}")
    print(f"  Number of factors: {len(model.factor_names)}")
    print(f"  Fit method: {model.fit_method}")
    print(f"  Variable selection: {model.variable_selection}")

    print("\nAccessing individual asset fit objects:")
    asset1_fit = model.asset_fit["Asset1"]
    print(f"  Asset1 regression object type: {type(asset1_fit).__name__}")
    print(f"  Asset1 parameters: {asset1_fit.params}")
    print(f"  Asset1 R²: {asset1_fit.rsquared:.4f}")
    print(f"  Asset1 p-values: {asset1_fit.pvalues}")

    print("\nResiduals shape:", model.residuals.shape)
    print("First 5 Asset1 residuals:", model.residuals[0, :5])


if __name__ == "__main__":
    print("\nfacmodts Package: Basic Usage Examples")
    print("=" * 80)

    # Run all examples
    example1_basic_ls()
    example2_dls()
    example3_excess_returns()
    example4_control_parameters()
    example5_model_inspection()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")
