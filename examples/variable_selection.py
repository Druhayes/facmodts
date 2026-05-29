"""
Variable selection examples for facmodts package.

This script demonstrates stepwise, best subsets, and LARS/Lasso variable
selection methods for factor model fitting.
"""

import numpy as np
import polars as pl
from datetime import date, timedelta

from facmodts import fit_tsfm


def create_data_with_sparse_factors(n_periods=150, n_important=2, n_noise=5):
    """
    Create return data where only a few factors matter (sparse structure).

    Parameters:
        n_periods: Number of time periods
        n_important: Number of truly important factors
        n_noise: Number of noise factors

    Returns:
        Polars DataFrame with asset/factor returns.
        Dictionary with true important factor indices.
    """
    # Generate dates
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_periods)]

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate important factors
    important_factors = {}
    for i in range(n_important):
        factor = 0.005 + 0.01 * np.sin(np.linspace(0, 4 * np.pi, n_periods)) + 0.008 * np.random.randn(n_periods)
        important_factors[f"Factor{i+1}"] = factor

    # Generate noise factors (no real predictive power)
    noise_factors = {}
    for i in range(n_noise):
        factor = 0.008 * np.random.randn(n_periods)
        noise_factors[f"NoiseFactor{i+1}"] = factor

    # Asset depends only on important factors
    # True betas: [0.8, 0.4] for important factors, 0 for noise
    asset = 0.001
    for i, (name, values) in enumerate(important_factors.items()):
        beta = [0.8, 0.4][i]  # Different loadings
        asset += beta * values

    asset += 0.005 * np.random.randn(n_periods)  # Idiosyncratic risk

    # Combine all data
    data_dict = {"date": dates, "Asset": asset}
    data_dict.update(important_factors)
    data_dict.update(noise_factors)

    data = pl.DataFrame(data_dict)

    return data, list(important_factors.keys())


def example1_stepwise_vs_none():
    """Example 1: Compare stepwise selection to using all factors."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Stepwise Selection vs All Factors")
    print("="*80)

    # Create sparse data (2 important + 5 noise = 7 total factors)
    data, important = create_data_with_sparse_factors(n_periods=150)
    all_factors = [col for col in data.columns if col.startswith("Factor") or col.startswith("Noise")]

    print(f"\nData: {len(data)} observations")
    print(f"Factors: {len(all_factors)} total ({len(important)} important)")
    print(f"Important factors: {important}")

    # Fit with all factors (no selection)
    model_all = fit_tsfm(
        data=data,
        asset_names=["Asset"],
        factor_names=all_factors,
        fit_method="LS",
        variable_selection="none"
    )

    # Fit with stepwise selection (BIC)
    model_step = fit_tsfm(
        data=data,
        asset_names=["Asset"],
        factor_names=all_factors,
        fit_method="LS",
        variable_selection="stepwise",
        step_direction="both",
        step_criterion="bic"
    )

    print("\n" + "-"*80)
    print(f"{'Method':<20} {'N Factors':<12} {'R²':<10} {'Top Factors':<40}")
    print("-"*80)

    # All factors
    n_all = len(all_factors)
    top_all_idx = np.argsort(np.abs(model_all.beta[0, :]))[-3:]  # Top 3 by magnitude
    top_all = [all_factors[i] for i in top_all_idx]
    print(f"{'All factors':<20} {n_all:<12} {model_all.r_squared[0]:<10.4f} {', '.join(top_all):<40}")

    # Stepwise
    n_step = np.sum(model_step.beta[0, :] != 0)
    selected_idx = np.where(model_step.beta[0, :] != 0)[0]
    selected = [all_factors[i] for i in selected_idx]
    print(f"{'Stepwise (BIC)':<20} {n_step:<12} {model_step.r_squared[0]:<10.4f} {', '.join(selected):<40}")

    print("\nNote: Stepwise should select mostly the important factors,")
    print("      achieving similar R² with fewer factors (parsimony).")


def example2_stepwise_directions():
    """Example 2: Compare stepwise directions (forward, backward, both)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Stepwise Directions (Forward, Backward, Both)")
    print("="*80)

    data, important = create_data_with_sparse_factors(n_periods=120)
    all_factors = [col for col in data.columns if col.startswith("Factor") or col.startswith("Noise")]

    directions = ["forward", "backward", "both"]

    print(f"\nFitting with different stepwise directions:")
    print("-"*70)
    print(f"{'Direction':<15} {'N Selected':<12} {'R²':<10} {'Selected Factors':<30}")
    print("-"*70)

    for direction in directions:
        model = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=all_factors,
            variable_selection="stepwise",
            step_direction=direction,
            step_criterion="bic"
        )

        n_selected = np.sum(model.beta[0, :] != 0)
        selected_idx = np.where(model.beta[0, :] != 0)[0]
        selected = [all_factors[i] for i in selected_idx]

        print(f"{direction:<15} {n_selected:<12} {model.r_squared[0]:<10.4f} {', '.join(selected):<30}")

    print("\nNote: All directions should converge to similar final models,")
    print("      though the search path may differ.")


def example3_best_subsets():
    """Example 3: Best subset selection with different subset sizes."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Best Subset Selection")
    print("="*80)

    data, important = create_data_with_sparse_factors(n_periods=120, n_important=2, n_noise=4)
    all_factors = [col for col in data.columns if col.startswith("Factor") or col.startswith("Noise")]

    print(f"\nTrying different subset sizes (nvmin to nvmax):")
    print("-"*70)
    print(f"{'Subset Size':<15} {'N Selected':<12} {'R²':<10} {'Selected Factors':<30}")
    print("-"*70)

    subset_configs = [
        (1, 1, "Size 1 only"),
        (2, 2, "Size 2 only"),
        (3, 3, "Size 3 only"),
        (1, 3, "Best in [1,3]"),
    ]

    for nvmin, nvmax, label in subset_configs:
        model = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=all_factors,
            variable_selection="subsets",
            nvmin=nvmin,
            nvmax=nvmax
        )

        n_selected = np.sum(model.beta[0, :] != 0)
        selected_idx = np.where(model.beta[0, :] != 0)[0]
        selected = [all_factors[i] for i in selected_idx]

        print(f"{label:<15} {n_selected:<12} {model.r_squared[0]:<10.4f} {', '.join(selected):<30}")

    print("\nNote: Best subsets exhaustively searches all combinations,")
    print("      guaranteeing the optimal subset of each size.")


def example4_lars_criteria():
    """Example 4: LARS with different model selection criteria."""
    print("\n" + "="*80)
    print("EXAMPLE 4: LARS/Lasso with Different Criteria")
    print("="*80)

    data, important = create_data_with_sparse_factors(n_periods=150, n_important=2, n_noise=5)
    all_factors = [col for col in data.columns if col.startswith("Factor") or col.startswith("Noise")]

    criteria = ["cp", "aic", "bic"]

    print(f"\nLARS with different selection criteria:")
    print("-"*70)
    print(f"{'Criterion':<15} {'N Selected':<12} {'R²':<10} {'Selected Factors':<30}")
    print("-"*70)

    for criterion in criteria:
        model = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=all_factors,
            variable_selection="lars",
            lars_criterion=criterion
        )

        n_selected = np.sum(model.beta[0, :] != 0)
        selected_idx = np.where(model.beta[0, :] != 0)[0]
        selected = [all_factors[i] for i in selected_idx]

        print(f"{criterion:<15} {n_selected:<12} {model.r_squared[0]:<10.4f} {', '.join(selected):<30}")

    print("\nNote: BIC tends to select fewer factors (more parsimonious),")
    print("      AIC may select more, Cp is a compromise.")
    print("      All produce sparse solutions (many zero coefficients).")


def example5_selection_with_robust():
    """Example 5: Variable selection combined with robust regression."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Variable Selection + Robust Regression")
    print("="*80)

    # Create data with outliers
    data, important = create_data_with_sparse_factors(n_periods=150, n_important=2, n_noise=5)

    # Add outliers to asset returns
    outlier_idx = np.random.choice(150, size=8, replace=False)
    asset_values = data["Asset"].to_numpy().copy()  # Make mutable copy
    asset_values[outlier_idx] += np.random.choice([-0.08, 0.08], size=8)
    data = data.with_columns(pl.Series("Asset", asset_values))

    all_factors = [col for col in data.columns if col.startswith("Factor") or col.startswith("Noise")]

    print(f"\nComparing selection methods with Robust vs LS:")
    print("-"*80)
    print(f"{'Method':<30} {'Fit Method':<12} {'N Selected':<12} {'R²':<10}")
    print("-"*80)

    # Stepwise + LS
    model_step_ls = fit_tsfm(
        data=data, asset_names=["Asset"], factor_names=all_factors,
        variable_selection="stepwise", fit_method="LS"
    )
    n_step_ls = np.sum(model_step_ls.beta[0, :] != 0)
    print(f"{'Stepwise':<30} {'LS':<12} {n_step_ls:<12} {model_step_ls.r_squared[0]:<10.4f}")

    # Stepwise + Robust
    model_step_rob = fit_tsfm(
        data=data, asset_names=["Asset"], factor_names=all_factors,
        variable_selection="stepwise", fit_method="Robust", family="bisquare"
    )
    n_step_rob = np.sum(model_step_rob.beta[0, :] != 0)
    print(f"{'Stepwise':<30} {'Robust':<12} {n_step_rob:<12} {model_step_rob.r_squared[0]:<10.4f}")

    # LARS (always uses its own algorithm)
    model_lars = fit_tsfm(
        data=data, asset_names=["Asset"], factor_names=all_factors,
        variable_selection="lars", lars_criterion="bic"
    )
    n_lars = np.sum(model_lars.beta[0, :] != 0)
    print(f"{'LARS':<30} {'LARS':<12} {n_lars:<12} {model_lars.r_squared[0]:<10.4f}")

    print("\nNote: Robust regression resists outliers during factor selection,")
    print("      potentially leading to different factor choices than LS.")


def example6_comparison_table():
    """Example 6: Comprehensive comparison of all methods."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Comprehensive Method Comparison")
    print("="*80)

    data, important = create_data_with_sparse_factors(n_periods=150, n_important=2, n_noise=6)
    all_factors = [col for col in data.columns if col.startswith("Factor") or col.startswith("Noise")]

    print(f"\nAll variable selection methods:")
    print(f"True important factors: {important}")
    print("-"*90)
    print(f"{'Method':<30} {'N Factors':<12} {'R²':<10} {'% Important':<15} {'Selected':<20}")
    print("-"*90)

    methods = [
        ("none", {}),
        ("stepwise-forward", {"variable_selection": "stepwise", "step_direction": "forward"}),
        ("stepwise-backward", {"variable_selection": "stepwise", "step_direction": "backward"}),
        ("stepwise-both-AIC", {"variable_selection": "stepwise", "step_criterion": "aic"}),
        ("stepwise-both-BIC", {"variable_selection": "stepwise", "step_criterion": "bic"}),
        ("subsets", {"variable_selection": "subsets", "nvmin": 1, "nvmax": 4}),
        ("lars-cp", {"variable_selection": "lars", "lars_criterion": "cp"}),
        ("lars-aic", {"variable_selection": "lars", "lars_criterion": "aic"}),
        ("lars-bic", {"variable_selection": "lars", "lars_criterion": "bic"}),
    ]

    for method_name, kwargs in methods:
        model = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=all_factors,
            **kwargs
        )

        n_selected = np.sum(model.beta[0, :] != 0)
        selected_idx = np.where(model.beta[0, :] != 0)[0]
        selected = [all_factors[i] for i in selected_idx]

        # Calculate % of selected that are important
        n_important_selected = sum(1 for f in selected if f in important)
        pct_important = (n_important_selected / len(selected) * 100) if selected else 0

        selected_str = ', '.join(selected[:3])  # Show first 3
        if len(selected) > 3:
            selected_str += "..."

        print(f"{method_name:<30} {n_selected:<12} {model.r_squared[0]:<10.4f} {pct_important:<14.1f}% {selected_str:<20}")

    print("\nConclusion: Selection methods balance parsimony and fit.")
    print("            BIC-based methods tend to select fewer factors.")
    print("            All should identify the truly important factors.")


if __name__ == "__main__":
    print("\nfacmodts Package: Variable Selection Examples")
    print("=" * 80)

    example1_stepwise_vs_none()
    example2_stepwise_directions()
    example3_best_subsets()
    example4_lars_criteria()
    example5_selection_with_robust()
    example6_comparison_table()

    print("\n" + "="*80)
    print("All variable selection examples completed!")
    print("="*80 + "\n")
