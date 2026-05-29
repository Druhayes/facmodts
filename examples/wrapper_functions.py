"""
Examples demonstrating convenience wrapper functions.

This script shows how to use the specialized wrappers for common factor models:
1. Market-timing models (Henriksson-Merton)
2. Up/down market models
3. Fama-French 3-factor models
4. Fama-French 4-factor (Carhart) models
"""

import numpy as np
import polars as pl
from datetime import date, timedelta

from facmodts import (
    fit_tsfm_mt,
    fit_tsfm_up_dn,
    fit_ff3_model,
    fit_ff4_model,
)


def create_fund_data(n_periods=120, seed=42):
    """
    Create sample fund returns with market exposure.

    Parameters
    ----------
    n_periods : int
        Number of time periods.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    polars.DataFrame
        Fund returns, market returns, and risk-free rate.
    """
    np.random.seed(seed)

    dates = [date(2020, 1, 1) + timedelta(days=i * 7) for i in range(n_periods)]

    # Market returns with some volatility
    market = 0.002 + 0.03 * np.sin(np.linspace(0, 4 * np.pi, n_periods)) + 0.02 * np.random.randn(n_periods)

    # Risk-free rate (constant 0.1% weekly)
    rf = np.ones(n_periods) * 0.001

    # Fund with market timing skill
    # High beta (1.2) in up markets, low beta (0.4) in down markets
    fund_mt = np.where(
        market >= rf,
        rf + 1.2 * (market - rf) + 0.005 * np.random.randn(n_periods),  # Up market
        rf + 0.4 * (market - rf) + 0.005 * np.random.randn(n_periods),  # Down market
    )

    # Standard long-only fund
    fund_long = rf + 0.9 * (market - rf) + 0.006 * np.random.randn(n_periods)

    data = pl.DataFrame({
        "date": dates,
        "MarketTimingFund": fund_mt,
        "LongOnlyFund": fund_long,
        "Market": market,
        "RF": rf,
    })

    return data


def create_ff_data(n_periods=120, seed=42):
    """
    Create sample Fama-French factor data with fund returns.

    Parameters
    ----------
    n_periods : int
        Number of time periods.
    seed : int
        Random seed.

    Returns
    -------
    polars.DataFrame
        FF factors and fund returns.
    """
    np.random.seed(seed)

    dates = [date(2020, 1, 1) + timedelta(days=i * 7) for i in range(n_periods)]

    # Fama-French factors (already excess returns)
    mkt = 0.003 + 0.025 * np.random.randn(n_periods)
    smb = 0.001 + 0.015 * np.random.randn(n_periods)
    hml = -0.0005 + 0.012 * np.random.randn(n_periods)
    mom = 0.002 + 0.018 * np.random.randn(n_periods)

    # Value fund: High HML loading
    value_fund = 0.001 + 0.8 * mkt + 0.2 * smb + 0.6 * hml + 0.008 * np.random.randn(n_periods)

    # Growth fund: Negative HML loading
    growth_fund = 0.002 + 1.0 * mkt - 0.1 * smb - 0.4 * hml + 0.01 * np.random.randn(n_periods)

    # Small-cap fund: High SMB loading
    small_cap_fund = 0.0015 + 0.9 * mkt + 0.7 * smb + 0.1 * hml + 0.012 * np.random.randn(n_periods)

    # Momentum fund: High MOM loading
    momentum_fund = 0.0025 + 0.85 * mkt + 0.1 * smb + 0.0 * hml + 0.5 * mom + 0.009 * np.random.randn(n_periods)

    data = pl.DataFrame({
        "date": dates,
        "MKT": mkt,
        "SMB": smb,
        "HML": hml,
        "MOM": mom,
        "ValueFund": value_fund,
        "GrowthFund": growth_fund,
        "SmallCapFund": small_cap_fund,
        "MomentumFund": momentum_fund,
    })

    return data


def example1_market_timing():
    """
    Example 1: Market-timing model (Henriksson-Merton).

    Demonstrates how to fit a market-timing model that adds a down-market
    factor to capture timing skills.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Market-Timing Model (Henriksson-Merton)")
    print("=" * 80)

    # Create sample data
    data = create_fund_data(n_periods=120)

    print("\nFitting market-timing model with LS...")
    print("Model includes:")
    print("  - Market excess returns")
    print("  - Down-market factor: max(0, R_f - R_m)")

    # Fit market-timing model
    mt_model = fit_tsfm_mt(
        asset_names=["MarketTimingFund"],
        mkt_name="Market",
        rf_name="RF",
        data=data,
        fit_method="LS",
    )

    print(f"\nFactors in model: {mt_model.factor_names}")
    print(f"Factor loadings (betas):")
    print(f"  Market:       {mt_model.beta[0, 0]:.4f}")
    print(f"  Down-market:  {mt_model.beta[0, 1]:.4f}")
    print(f"Alpha (intercept): {mt_model.alpha[0]:.6f}")
    print(f"R-squared:         {mt_model.r_squared[0]:.4f}")

    print("\nInterpretation:")
    if mt_model.beta[0, 1] > 0:
        print(f"  Positive down-market beta ({mt_model.beta[0, 1]:.4f}) suggests market-timing skill")
        print(f"  Manager benefits from {mt_model.beta[0, 1]:.4f} 'free put options' per unit exposure")
    else:
        print(f"  Negative down-market beta suggests poor market timing")

    # Compare to standard model (no timing)
    from facmodts import fit_tsfm

    std_model = fit_tsfm(
        asset_names=["MarketTimingFund"],
        factor_names=["Market"],
        rf_name="RF",
        data=data,
        fit_method="LS",
    )

    print(f"\nComparison to standard model:")
    print(f"  Standard model R²:       {std_model.r_squared[0]:.4f}")
    print(f"  Market-timing model R²:  {mt_model.r_squared[0]:.4f}")
    print(f"  Improvement:             {mt_model.r_squared[0] - std_model.r_squared[0]:.4f}")


def example2_up_down_markets():
    """
    Example 2: Up/down market models.

    Fit separate models for bull and bear markets to capture regime-dependent
    factor sensitivities.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Up/Down Market Models")
    print("=" * 80)

    # Create sample data
    data = create_fund_data(n_periods=120)

    print("\nFitting separate models for up and down markets...")
    print("  Up market:   periods where (Market - RF) >= 0")
    print("  Down market: periods where (Market - RF) < 0")

    # Fit up/down model
    updn_model = fit_tsfm_up_dn(
        asset_names=["MarketTimingFund", "LongOnlyFund"],
        mkt_name="Market",
        rf_name="RF",
        data=data,
        fit_method="LS",
    )

    # Count up/down periods
    n_up = np.sum(updn_model.up_market_flag)
    n_down = np.sum(~updn_model.up_market_flag)

    print(f"\nMarket regime split:")
    print(f"  Up market periods:   {n_up} ({n_up / len(updn_model.up_market_flag) * 100:.1f}%)")
    print(f"  Down market periods: {n_down} ({n_down / len(updn_model.up_market_flag) * 100:.1f}%)")

    print(f"\n{'Asset':<20} {'Up Beta':<12} {'Down Beta':<12} {'Difference':<12}")
    print("-" * 56)

    for i, asset in enumerate(updn_model.up_model.asset_names):
        beta_up = updn_model.up_model.beta[i, 0]
        beta_down = updn_model.down_model.beta[i, 0]
        diff = beta_up - beta_down

        print(f"{asset:<20} {beta_up:<12.4f} {beta_down:<12.4f} {diff:<12.4f}")

    print("\nInterpretation:")
    print("  Market-timing fund should have higher up-market beta (tactical overweighting)")
    print("  Long-only fund should have similar betas in both regimes")

    # Show R² for each regime
    print(f"\nModel fit (R²):")
    print(f"  {'Asset':<20} {'Up Market':<12} {'Down Market':<12}")
    print("  " + "-" * 44)
    for i, asset in enumerate(updn_model.up_model.asset_names):
        r2_up = updn_model.up_model.r_squared[i]
        r2_down = updn_model.down_model.r_squared[i]
        print(f"  {asset:<20} {r2_up:<12.4f} {r2_down:<12.4f}")


def example3_fama_french_3factor():
    """
    Example 3: Fama-French 3-factor model.

    Fit the classic FF3 model to analyze fund style exposures.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Fama-French 3-Factor Model")
    print("=" * 80)

    # Create FF data
    data = create_ff_data(n_periods=120)

    print("\nFitting Fama-French 3-factor model...")
    print("Factors:")
    print("  MKT - Market excess returns")
    print("  SMB - Small-minus-big (size factor)")
    print("  HML - High-minus-low (value factor)")

    # Fit FF3 model
    ff3_model = fit_ff3_model(
        asset_names=["ValueFund", "GrowthFund", "SmallCapFund"],
        factor_data=data,
        fit_method="LS",
    )

    print(f"\nFactor loadings:")
    print(f"{'Fund':<15} {'Alpha':>10} {'MKT':>10} {'SMB':>10} {'HML':>10} {'R²':>10}")
    print("-" * 65)

    for i, asset in enumerate(ff3_model.asset_names):
        alpha = ff3_model.alpha[i]
        mkt_beta = ff3_model.beta[i, 0]
        smb_beta = ff3_model.beta[i, 1]
        hml_beta = ff3_model.beta[i, 2]
        r2 = ff3_model.r_squared[i]

        print(
            f"{asset:<15} {alpha:>10.4f} {mkt_beta:>10.4f} {smb_beta:>10.4f} "
            f"{hml_beta:>10.4f} {r2:>10.4f}"
        )

    print("\nStyle classification:")
    print("  Value fund:     High HML (positive value tilt)")
    print("  Growth fund:    Low/negative HML (growth tilt)")
    print("  Small-cap fund: High SMB (small-cap exposure)")

    # Analyze alpha
    print(f"\nAlpha analysis (excess returns not explained by factors):")
    for i, asset in enumerate(ff3_model.asset_names):
        alpha = ff3_model.alpha[i]
        # Annualized alpha (assuming weekly data)
        alpha_annual = alpha * 52

        if abs(alpha_annual) < 0.01:
            interpretation = "no significant skill"
        elif alpha_annual > 0:
            interpretation = f"+{alpha_annual*100:.2f}% annual outperformance"
        else:
            interpretation = f"{alpha_annual*100:.2f}% annual underperformance"

        print(f"  {asset:<15}: {alpha:.6f} weekly ({interpretation})")


def example4_fama_french_4factor():
    """
    Example 4: Fama-French 4-factor (Carhart) model.

    Extend FF3 with momentum factor to capture momentum strategies.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Fama-French 4-Factor (Carhart) Model")
    print("=" * 80)

    # Create FF data
    data = create_ff_data(n_periods=120)

    print("\nFitting Fama-French 4-factor model...")
    print("Factors:")
    print("  MKT - Market excess returns")
    print("  SMB - Small-minus-big (size)")
    print("  HML - High-minus-low (value)")
    print("  MOM - Winners-minus-losers (momentum)")

    # Fit FF4 model
    ff4_model = fit_ff4_model(
        asset_names=["ValueFund", "GrowthFund", "MomentumFund"],
        factor_data=data,
        fit_method="LS",
    )

    print(f"\nFactor loadings:")
    print(f"{'Fund':<15} {'Alpha':>9} {'MKT':>9} {'SMB':>9} {'HML':>9} {'MOM':>9} {'R²':>9}")
    print("-" * 72)

    for i, asset in enumerate(ff4_model.asset_names):
        alpha = ff4_model.alpha[i]
        mkt_beta = ff4_model.beta[i, 0]
        smb_beta = ff4_model.beta[i, 1]
        hml_beta = ff4_model.beta[i, 2]
        mom_beta = ff4_model.beta[i, 3]
        r2 = ff4_model.r_squared[i]

        print(
            f"{asset:<15} {alpha:>9.5f} {mkt_beta:>9.4f} {smb_beta:>9.4f} "
            f"{hml_beta:>9.4f} {mom_beta:>9.4f} {r2:>9.4f}"
        )

    print("\nMomentum exposure analysis:")
    for i, asset in enumerate(ff4_model.asset_names):
        mom_beta = ff4_model.beta[i, 3]

        if abs(mom_beta) < 0.1:
            strategy = "neutral momentum strategy"
        elif mom_beta > 0.3:
            strategy = "strong momentum tilt (buys winners)"
        elif mom_beta > 0:
            strategy = "moderate momentum tilt"
        elif mom_beta > -0.3:
            strategy = "contrarian tilt (buys losers)"
        else:
            strategy = "strong contrarian tilt"

        print(f"  {asset:<15}: MOM = {mom_beta:>6.3f}  ({strategy})")


def example5_ff3_vs_ff4():
    """
    Example 5: Compare FF3 vs FF4 models.

    Show how adding momentum improves model fit.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: FF3 vs FF4 Model Comparison")
    print("=" * 80)

    # Create FF data
    data = create_ff_data(n_periods=120)

    print("\nComparing 3-factor and 4-factor models for momentum fund...")

    # Fit both models
    ff3_model = fit_ff3_model(
        asset_names=["MomentumFund"],
        factor_data=data,
        fit_method="LS",
    )

    ff4_model = fit_ff4_model(
        asset_names=["MomentumFund"],
        factor_data=data,
        fit_method="LS",
    )

    print(f"\nFF3 Model (3 factors):")
    print(f"  Alpha:      {ff3_model.alpha[0]:.6f}")
    print(f"  MKT beta:   {ff3_model.beta[0, 0]:.4f}")
    print(f"  SMB beta:   {ff3_model.beta[0, 1]:.4f}")
    print(f"  HML beta:   {ff3_model.beta[0, 2]:.4f}")
    print(f"  R²:         {ff3_model.r_squared[0]:.4f}")
    print(f"  Resid SD:   {ff3_model.resid_sd[0]:.6f}")

    print(f"\nFF4 Model (4 factors, adds momentum):")
    print(f"  Alpha:      {ff4_model.alpha[0]:.6f}")
    print(f"  MKT beta:   {ff4_model.beta[0, 0]:.4f}")
    print(f"  SMB beta:   {ff4_model.beta[0, 1]:.4f}")
    print(f"  HML beta:   {ff4_model.beta[0, 2]:.4f}")
    print(f"  MOM beta:   {ff4_model.beta[0, 3]:.4f}")
    print(f"  R²:         {ff4_model.r_squared[0]:.4f}")
    print(f"  Resid SD:   {ff4_model.resid_sd[0]:.6f}")

    print(f"\nModel comparison:")
    r2_improvement = ff4_model.r_squared[0] - ff3_model.r_squared[0]
    resid_reduction = (ff3_model.resid_sd[0] - ff4_model.resid_sd[0]) / ff3_model.resid_sd[0] * 100

    print(f"  R² improvement:        {r2_improvement:.4f} ({r2_improvement/ff3_model.r_squared[0]*100:.1f}%)")
    print(f"  Residual SD reduction: {resid_reduction:.2f}%")

    if r2_improvement > 0.05:
        print("\n  → Momentum factor significantly improves model fit")
        print("  → This fund likely employs momentum strategies")
    else:
        print("\n  → Minimal improvement from momentum factor")
        print("  → Fund may not use significant momentum strategies")


def example6_robust_wrappers():
    """
    Example 6: Robust estimation with wrappers.

    Show how to use robust regression with wrapper functions.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Robust Estimation with Wrappers")
    print("=" * 80)

    # Create FF data
    data = create_ff_data(n_periods=120)

    # Add outliers to one fund
    outlier_idx = [10, 25, 50, 75, 100]
    growth_returns = data["GrowthFund"].to_numpy().copy()
    growth_returns[outlier_idx] += np.random.choice([-0.15, 0.15], size=len(outlier_idx))
    data = data.with_columns(pl.Series("GrowthFund", growth_returns))

    print("\nAdded 5 outliers to GrowthFund returns")
    print("Comparing LS vs Robust estimation...")

    # Fit with LS
    ff3_ls = fit_ff3_model(
        asset_names=["GrowthFund"],
        factor_data=data,
        fit_method="LS",
    )

    # Fit with Robust
    ff3_robust = fit_ff3_model(
        asset_names=["GrowthFund"],
        factor_data=data,
        fit_method="Robust",
        family="bisquare",
    )

    print(f"\n{'Method':<10} {'MKT':>10} {'SMB':>10} {'HML':>10} {'R²':>10} {'Resid SD':>12}")
    print("-" * 62)

    print(
        f"{'LS':<10} {ff3_ls.beta[0, 0]:>10.4f} {ff3_ls.beta[0, 1]:>10.4f} "
        f"{ff3_ls.beta[0, 2]:>10.4f} {ff3_ls.r_squared[0]:>10.4f} {ff3_ls.resid_sd[0]:>12.6f}"
    )

    print(
        f"{'Robust':<10} {ff3_robust.beta[0, 0]:>10.4f} {ff3_robust.beta[0, 1]:>10.4f} "
        f"{ff3_robust.beta[0, 2]:>10.4f} {ff3_robust.r_squared[0]:>10.4f} "
        f"{ff3_robust.resid_sd[0]:>12.6f}"
    )

    print("\nBeta differences (Robust - LS):")
    beta_diff = ff3_robust.beta[0, :] - ff3_ls.beta[0, :]
    print(f"  MKT: {beta_diff[0]:+.4f}")
    print(f"  SMB: {beta_diff[1]:+.4f}")
    print(f"  HML: {beta_diff[2]:+.4f}")

    print("\nNote: Robust estimates are less affected by the 5 outliers,")
    print("      providing more reliable factor exposures.")


if __name__ == "__main__":
    print("\nfacmodts Package: Wrapper Function Examples")
    print("=" * 80)

    example1_market_timing()
    example2_up_down_markets()
    example3_fama_french_3factor()
    example4_fama_french_4factor()
    example5_ff3_vs_ff4()
    example6_robust_wrappers()

    print("\n" + "=" * 80)
    print("All wrapper function examples completed!")
    print("=" * 80 + "\n")
