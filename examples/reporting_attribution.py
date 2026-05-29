"""
Examples demonstrating reporting and performance attribution.

This script shows how to:
1. Print model summaries
2. Generate detailed statistical summaries
3. Make predictions
4. Compute performance attribution
5. Analyze factor contributions to returns
"""

import numpy as np
import polars as pl
from datetime import date, timedelta

from facmodts import (
    fit_tsfm,
    pa_fm,
    summary_tsfm,
    print_summary_tsfm,
    print_tsfm,
    predict_tsfm,
    print_pafm,
    summary_pafm,
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


def example1_print_model():
    """
    Example 1: Print model summary.

    Demonstrates print_tsfm() for quick model overview.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Print Model Summary")
    print("=" * 80)

    data = create_portfolio_data(n_periods=100)

    # Fit model
    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    # Print summary
    print("\nBasic model print:")
    print_tsfm(model)


def example2_detailed_summary():
    """
    Example 2: Detailed statistical summary.

    Demonstrates summary_tsfm() for coefficient inference.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Detailed Statistical Summary")
    print("=" * 80)

    data = create_portfolio_data(n_periods=100)

    # Fit model
    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund", "BalancedFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    # Generate summary
    summary = summary_tsfm(model)

    # Print detailed summary
    print("\nDetailed summary with t-statistics and p-values:")
    print_summary_tsfm(summary, digits=4)

    # Extract information programmatically
    print("\n" + "-" * 80)
    print("Programmatic access to summary information:")
    print("-" * 80)

    for asset_summary in summary["summaries"]:
        asset = asset_summary["asset"]
        coefs = asset_summary["coefficients"]
        pvals = asset_summary["p_values"]

        # Find significant factors (p < 0.05)
        sig_factors = []
        if pvals is not None:
            for j, (name, p) in enumerate(zip(asset_summary["coef_names"][1:], pvals[1:])):
                if p < 0.05:
                    sig_factors.append((name, coefs[j+1], p))

        print(f"\n{asset}:")
        print(f"  R²: {asset_summary['r_squared']:.4f}")
        print(f"  Significant factors (p < 0.05):")
        for name, coef, p in sig_factors:
            print(f"    {name:<10}: β = {coef:>7.4f}  (p = {p:.4f})")


def example3_predictions():
    """
    Example 3: In-sample and out-of-sample predictions.

    Demonstrates predict_tsfm() for forecasting.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Model Predictions")
    print("=" * 80)

    data = create_portfolio_data(n_periods=100)

    # Fit model on first 80 periods
    train_data = data[:80]

    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund"],
        factor_names=["Market", "Size", "Value"],
        data=train_data,
        fit_method="LS",
    )

    # In-sample fitted values
    fitted = predict_tsfm(model)
    print(f"\nIn-sample fitted values shape: {fitted.shape}")
    print(f"First 5 fitted values for GrowthFund: {fitted[0, :5]}")

    # Out-of-sample predictions (last 20 periods)
    test_factors = data[80:].select(["Market", "Size", "Value"])
    predictions = predict_tsfm(model, newdata=test_factors)

    print(f"\nOut-of-sample predictions shape: {predictions.shape}")

    # Calculate prediction errors
    actual_growth = data["GrowthFund"][80:].to_numpy()
    pred_growth = predictions[0, :]
    errors = actual_growth - pred_growth

    print(f"\nPrediction quality (GrowthFund):")
    print(f"  Mean absolute error: {np.mean(np.abs(errors)):.5f}")
    print(f"  Root mean squared error: {np.sqrt(np.mean(errors**2)):.5f}")


def example4_performance_attribution():
    """
    Example 4: Performance attribution.

    Decomposes returns into factor contributions.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Performance Attribution")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    # Fit model
    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund", "BalancedFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    # Compute attribution
    pa = pa_fm(model)

    # Print attribution
    print("\nPerformance attribution summary:")
    print_pafm(pa)

    print("\n" + "-" * 80)
    print("Interpretation:")
    print("-" * 80)

    # Analyze factor contributions
    for i, asset in enumerate(pa.asset_names):
        print(f"\n{asset}:")

        # Cumulative attributions
        cum_market = pa.cum_ret_attr_f[i, 0]
        cum_size = pa.cum_ret_attr_f[i, 1]
        cum_value = pa.cum_ret_attr_f[i, 2]
        cum_specific = pa.cum_spec_ret[i]

        total_cum = cum_market + cum_size + cum_value + cum_specific

        print(f"  Cumulative return breakdown:")
        print(f"    Market:   {cum_market:>8.2%}  ({cum_market/total_cum*100:>5.1f}% of total)")
        print(f"    Size:     {cum_size:>8.2%}  ({cum_size/total_cum*100:>5.1f}% of total)")
        print(f"    Value:    {cum_value:>8.2%}  ({cum_value/total_cum*100:>5.1f}% of total)")
        print(f"    Specific: {cum_specific:>8.2%}  ({cum_specific/total_cum*100:>5.1f}% of total)")
        print(f"    Total:    {total_cum:>8.2%}")


def example5_time_series_attribution():
    """
    Example 5: Time series of factor attributions.

    Analyzes how factor contributions evolve over time.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Time Series Attribution Analysis")
    print("=" * 80)

    data = create_portfolio_data(n_periods=120)

    # Fit model
    model = fit_tsfm(
        asset_names=["GrowthFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    # Compute attribution
    pa = pa_fm(model)

    # Get time series for Growth Fund
    attr_ts = pa.attr_list["GrowthFund"]

    print("\nFirst 10 periods of factor attributions (GrowthFund):")
    print(attr_ts.head(10))

    # Compute summary statistics
    print("\nSummary statistics of factor attributions:")
    print(f"\n{'Factor':<15} {'Mean':>12} {'Std Dev':>12} {'Sharpe':>12}")
    print("-" * 52)

    for factor in pa.factor_names:
        mean = attr_ts[factor].mean()
        std = attr_ts[factor].std()
        sharpe = mean / std if std > 0 else 0

        print(f"{factor:<15} {mean:>12.5f} {std:>12.5f} {sharpe:>12.3f}")

    # Specific returns
    spec_mean = attr_ts["specific_returns"].mean()
    spec_std = attr_ts["specific_returns"].std()
    spec_sharpe = spec_mean / spec_std if spec_std > 0 else 0

    print(f"{'Specific':<15} {spec_mean:>12.5f} {spec_std:>12.5f} {spec_sharpe:>12.3f}")

    # Correlations between factor attributions
    print("\nCorrelations between factor attributions:")

    corr_data = attr_ts.select(pa.factor_names + ["specific_returns"]).to_pandas()
    corr_matrix = corr_data.corr()

    print(corr_matrix)


def example6_attribution_with_robust():
    """
    Example 6: Performance attribution with robust regression.

    Shows how outliers affect attribution.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Attribution with Robust vs LS")
    print("=" * 80)

    # Create data with outliers
    data = create_portfolio_data(n_periods=100)

    # Add outliers to GrowthFund
    growth_returns = data["GrowthFund"].to_numpy().copy()
    growth_returns[[10, 30, 50, 70, 90]] += np.array([0.15, -0.15, 0.15, -0.15, 0.15])
    data = data.with_columns(pl.Series("GrowthFund", growth_returns))

    print("\nAdded 5 large outliers to GrowthFund")

    # Fit with LS
    model_ls = fit_tsfm(
        asset_names=["GrowthFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    # Fit with Robust
    model_robust = fit_tsfm(
        asset_names=["GrowthFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="Robust",
        family="bisquare",
    )

    # Attribution for both
    pa_ls = pa_fm(model_ls)
    pa_robust = pa_fm(model_robust)

    print("\nComparison of cumulative factor attributions:")
    print(f"\n{'Method':<10} {'Market':>12} {'Size':>12} {'Value':>12} {'Specific':>12}")
    print("-" * 60)

    # LS
    print(f"{'LS':<10} {pa_ls.cum_ret_attr_f[0, 0]:>12.4f} "
          f"{pa_ls.cum_ret_attr_f[0, 1]:>12.4f} {pa_ls.cum_ret_attr_f[0, 2]:>12.4f} "
          f"{pa_ls.cum_spec_ret[0]:>12.4f}")

    # Robust
    print(f"{'Robust':<10} {pa_robust.cum_ret_attr_f[0, 0]:>12.4f} "
          f"{pa_robust.cum_ret_attr_f[0, 1]:>12.4f} {pa_robust.cum_ret_attr_f[0, 2]:>12.4f} "
          f"{pa_robust.cum_spec_ret[0]:>12.4f}")

    print("\nNote: Robust attribution is less affected by the 5 outliers,")
    print("      providing more reliable factor contribution estimates.")


def example7_complete_workflow():
    """
    Example 7: Complete analysis workflow.

    End-to-end example from fitting to reporting.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Complete Analysis Workflow")
    print("=" * 80)

    # Create data
    data = create_portfolio_data(n_periods=120)

    print("\n1. Fit model...")
    model = fit_tsfm(
        asset_names=["GrowthFund", "ValueFund", "BalancedFund"],
        factor_names=["Market", "Size", "Value"],
        data=data,
        fit_method="LS",
    )

    print("\n2. Quick model overview:")
    print_tsfm(model, digits=4)

    print("\n3. Detailed summary (ValueFund only):")
    summary = summary_tsfm(model)
    value_summary = summary["summaries"][1]  # ValueFund is second

    print(f"\nValueFund coefficient table:")
    print(f"{'Coefficient':<15} {'Estimate':>12} {'t-stat':>10} {'p-value':>10}")
    print("-" * 50)
    for j, name in enumerate(value_summary["coef_names"]):
        est = value_summary["coefficients"][j]
        t = value_summary["t_stats"][j]
        p = value_summary["p_values"][j]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{name:<15} {est:>12.4f} {t:>10.3f} {p:>10.4f} {sig}")

    print(f"\n4. Performance attribution:")
    pa = pa_fm(model)

    print(f"\nMean weekly factor contributions:")
    summary_pafm(pa, digits=5)

    print(f"\n5. Out-of-sample prediction:")
    # Predict next 4 weeks with hypothetical factor values
    future_factors = pl.DataFrame({
        "Market": [0.005, 0.010, -0.005, 0.008],
        "Size": [0.002, -0.001, 0.003, 0.001],
        "Value": [-0.001, 0.002, 0.001, -0.002],
    })

    predictions = predict_tsfm(model, newdata=future_factors)

    print(f"\nPredicted returns for next 4 weeks:")
    print(f"{'Fund':<15} {'Week 1':>10} {'Week 2':>10} {'Week 3':>10} {'Week 4':>10}")
    print("-" * 57)
    for i, asset in enumerate(model.asset_names):
        print(f"{asset:<15} ", end="")
        for week in range(4):
            print(f"{predictions[i, week]:>10.4f} ", end="")
        print()


if __name__ == "__main__":
    print("\nfacmodts Package: Reporting & Attribution Examples")
    print("=" * 80)

    example1_print_model()
    example2_detailed_summary()
    example3_predictions()
    example4_performance_attribution()
    example5_time_series_attribution()
    example6_attribution_with_robust()
    example7_complete_workflow()

    print("\n" + "=" * 80)
    print("All reporting and attribution examples completed!")
    print("=" * 80 + "\n")
