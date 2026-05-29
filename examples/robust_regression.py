"""
Robust regression examples for facmodts package.

This script demonstrates robust regression functionality, showing how
robust methods handle outliers better than standard LS regression.
"""

import numpy as np
import polars as pl
from datetime import date, timedelta

from facmodts import fit_tsfm


def create_data_with_outliers(n_periods=150, outlier_fraction=0.05):
    """
    Create return data with outliers to demonstrate robust regression.

    Parameters:
        n_periods: Number of time periods
        outlier_fraction: Fraction of data points to make outliers (default 5%)

    Returns:
        Polars DataFrame with asset/factor returns including outliers.
    """
    # Generate dates
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_periods)]

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate clean factor returns
    factor1 = 0.005 + 0.015 * np.sin(np.linspace(0, 8 * np.pi, n_periods)) + 0.01 * np.random.randn(n_periods)
    factor2 = -0.003 + 0.012 * np.cos(np.linspace(0, 6 * np.pi, n_periods)) + 0.008 * np.random.randn(n_periods)

    # Generate asset returns with known factor loadings
    # True betas: Asset1 = [1.0, 0.3], Asset2 = [0.7, -0.5]
    asset1 = 0.001 + 1.0 * factor1 + 0.3 * factor2 + 0.008 * np.random.randn(n_periods)
    asset2 = 0.002 + 0.7 * factor1 - 0.5 * factor2 + 0.010 * np.random.randn(n_periods)

    # Add outliers (large shocks on random days)
    n_outliers = int(n_periods * outlier_fraction)
    outlier_idx_1 = np.random.choice(n_periods, size=n_outliers, replace=False)
    outlier_idx_2 = np.random.choice(n_periods, size=n_outliers, replace=False)

    asset1[outlier_idx_1] += np.random.choice([-0.08, 0.08], size=n_outliers)
    asset2[outlier_idx_2] += np.random.choice([-0.10, 0.10], size=n_outliers)

    data = pl.DataFrame({
        "date": dates,
        "Asset1": asset1,
        "Asset2": asset2,
        "Factor1": factor1,
        "Factor2": factor2,
    })

    return data, {"Asset1": [1.0, 0.3], "Asset2": [0.7, -0.5]}


def example1_robust_vs_ls():
    """Example 1: Compare Robust and LS regression on data with outliers."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Robust vs LS Regression with Outliers")
    print("="*80)

    # Create data with 5% outliers
    data, true_betas = create_data_with_outliers(outlier_fraction=0.05)

    print(f"\nData: {len(data)} observations with ~5% outliers")
    print(f"True beta coefficients:")
    for asset, betas in true_betas.items():
        print(f"  {asset}: {betas}")

    # Fit LS model (affected by outliers)
    model_ls = fit_tsfm(
        data=data,
        asset_names=["Asset1", "Asset2"],
        factor_names=["Factor1", "Factor2"],
        fit_method="LS"
    )

    # Fit Robust model (resistant to outliers)
    model_robust = fit_tsfm(
        data=data,
        asset_names=["Asset1", "Asset2"],
        factor_names=["Factor1", "Factor2"],
        fit_method="Robust",
        family="bisquare"  # Tukey's bisquare (default)
    )

    # Compare results
    print("\n" + "-"*80)
    print(f"{'Method':<15} {'Asset':<10} {'Factor1 Beta':<15} {'Factor2 Beta':<15} {'Error':<10}")
    print("-"*80)

    for i, asset in enumerate(["Asset1", "Asset2"]):
        # LS results
        ls_err = np.sqrt(
            (model_ls.beta[i, 0] - true_betas[asset][0])**2 +
            (model_ls.beta[i, 1] - true_betas[asset][1])**2
        )
        print(f"{'LS':<15} {asset:<10} {model_ls.beta[i, 0]:>14.4f} {model_ls.beta[i, 1]:>14.4f} {ls_err:>9.4f}")

        # Robust results
        rob_err = np.sqrt(
            (model_robust.beta[i, 0] - true_betas[asset][0])**2 +
            (model_robust.beta[i, 1] - true_betas[asset][1])**2
        )
        print(f"{'Robust':<15} {asset:<10} {model_robust.beta[i, 0]:>14.4f} {model_robust.beta[i, 1]:>14.4f} {rob_err:>9.4f}")

        # True values
        print(f"{'True':<15} {asset:<10} {true_betas[asset][0]:>14.4f} {true_betas[asset][1]:>14.4f} {'0.0000':>9}")
        print()

    print("\nNote: Robust regression coefficients are typically closer to true values")
    print("      when outliers are present, demonstrating outlier resistance.")


def example2_robust_families():
    """Example 2: Compare different robust loss function families."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Different Robust Loss Functions")
    print("="*80)

    data, _ = create_data_with_outliers(outlier_fraction=0.08)  # More outliers

    families = ["bisquare", "huber", "hampel", "andrews"]

    print(f"\nFitting Asset1 with different robust methods:")
    print(f"Data has ~8% outliers")
    print("\n" + "-"*60)
    print(f"{'Family':<15} {'Factor1 Beta':<15} {'Factor2 Beta':<15} {'R²':<10}")
    print("-"*60)

    for family in families:
        model = fit_tsfm(
            data=data,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="Robust",
            family=family
        )

        print(f"{family:<15} {model.beta[0, 0]:>14.4f} {model.beta[0, 1]:>14.4f} {model.r_squared[0]:>9.4f}")

    print("\nLoss Function Characteristics:")
    print("  - bisquare: Completely rejects large outliers (most robust)")
    print("  - huber: Downweights outliers linearly (moderate robustness)")
    print("  - hampel: Three-part function (customizable)")
    print("  - andrews: Sinusoidal function (moderate robustness)")


def example3_tuning_constants():
    """Example 3: Effect of tuning constants on robustness."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Tuning Constants")
    print("="*80)

    data, true_beta = create_data_with_outliers(outlier_fraction=0.10)

    print("\nBisquare with different tuning constants:")
    print("(Higher tuning constant = more robust but less efficient)")
    print("\n" + "-"*70)
    print(f"{'Tuning Const':<15} {'Factor1 Beta':<15} {'Factor2 Beta':<15} {'Error':<10}")
    print("-"*70)

    tuning_values = [3.0, 4.685, 6.0, 8.0]  # 4.685 is default (95% efficiency)

    for tuning in tuning_values:
        model = fit_tsfm(
            data=data,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="Robust",
            family="bisquare",
            tuning_psi=tuning
        )

        error = np.sqrt(
            (model.beta[0, 0] - true_beta["Asset1"][0])**2 +
            (model.beta[0, 1] - true_beta["Asset1"][1])**2
        )

        label = f"{tuning:.2f}"
        if tuning == 4.685:
            label += " (default)"

        print(f"{label:<15} {model.beta[0, 0]:>14.4f} {model.beta[0, 1]:>14.4f} {error:>9.4f}")

    print(f"\n{'True values:':<15} {true_beta['Asset1'][0]:>14.4f} {true_beta['Asset1'][1]:>14.4f}")


def example4_severe_outliers():
    """Example 4: Performance with severe outliers (10% contamination)."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Severe Outlier Contamination (10%)")
    print("="*80)

    data, true_beta = create_data_with_outliers(outlier_fraction=0.10)

    # Fit all three methods
    model_ls = fit_tsfm(
        data=data,
        asset_names=["Asset1"],
        factor_names=["Factor1", "Factor2"],
        fit_method="LS"
    )

    model_dls = fit_tsfm(
        data=data,
        asset_names=["Asset1"],
        factor_names=["Factor1", "Factor2"],
        fit_method="DLS",
        decay=0.95
    )

    model_robust = fit_tsfm(
        data=data,
        asset_names=["Asset1"],
        factor_names=["Factor1", "Factor2"],
        fit_method="Robust",
        family="bisquare"
    )

    print("\nComparison with 10% outlier contamination:")
    print("-"*70)
    print(f"{'Method':<15} {'Factor1 Beta':<15} {'Factor2 Beta':<15} {'Error':<10}")
    print("-"*70)

    for name, model in [("LS", model_ls), ("DLS", model_dls), ("Robust", model_robust)]:
        error = np.sqrt(
            (model.beta[0, 0] - true_beta["Asset1"][0])**2 +
            (model.beta[0, 1] - true_beta["Asset1"][1])**2
        )
        print(f"{name:<15} {model.beta[0, 0]:>14.4f} {model.beta[0, 1]:>14.4f} {error:>9.4f}")

    print(f"{'True':<15} {true_beta['Asset1'][0]:>14.4f} {true_beta['Asset1'][1]:>14.4f} {'0.0000':>9}")

    print("\nConclusion: Robust regression maintains accuracy even with")
    print("            severe outlier contamination (10% of data).")


if __name__ == "__main__":
    print("\nfacmodts Package: Robust Regression Examples")
    print("=" * 80)

    example1_robust_vs_ls()
    example2_robust_families()
    example3_tuning_constants()
    example4_severe_outliers()

    print("\n" + "="*80)
    print("All robust regression examples completed!")
    print("="*80 + "\n")
