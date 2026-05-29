"""
Tests for robust regression functionality.

Tests cover:
- Robust regression with different loss functions
- Comparison with LS regression
- Outlier resistance
- Robust R-squared computation
"""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from facmodts import fit_tsfm, fit_rlm, compute_robust_scale, robust_r_squared


class TestRobustRegression:
    """Tests for basic robust regression functionality."""

    def test_robust_basic(self, synthetic_returns_simple):
        """Test basic robust regression."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="Robust"
        )

        # Check model structure
        assert model.fit_method == "Robust"
        assert len(model.asset_names) == 1
        assert len(model.factor_names) == 2

        # Check coefficient shapes
        assert model.alpha.shape == (1,)
        assert model.beta.shape == (1, 2)

        # R-squared should be reasonable (robust R²)
        assert 0 <= model.r_squared[0] <= 1

    def test_robust_multiple_assets(self, synthetic_returns_simple):
        """Test robust regression with multiple assets."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2", "Asset3"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="Robust"
        )

        # Check shapes
        assert model.alpha.shape == (3,)
        assert model.beta.shape == (3, 3)
        assert len(model.r_squared) == 3

        # All R-squared should be reasonable
        assert np.all((model.r_squared >= 0) & (model.r_squared <= 1))

    def test_robust_families(self, synthetic_returns_simple):
        """Test different robust loss function families."""
        families = ["bisquare", "huber", "hampel", "andrews"]

        for family in families:
            model = fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["Asset1"],
                factor_names=["Factor1"],
                fit_method="Robust",
                family=family
            )

            # Model should fit successfully
            assert model.r_squared[0] > 0
            assert model.resid_sd[0] > 0

    def test_robust_vs_ls_clean_data(self, synthetic_returns_simple):
        """Compare robust and LS on clean data (should be similar)."""
        model_ls = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        model_robust = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="Robust"
        )

        # Coefficients should be similar on clean data
        np.testing.assert_allclose(
            model_ls.beta,
            model_robust.beta,
            rtol=0.1,  # 10% tolerance
            atol=0.05
        )


class TestRobustOutlierResistance:
    """Test that robust regression is resistant to outliers."""

    def test_outlier_resistance(self):
        """Verify robust regression handles outliers better than LS."""
        # Create data with outliers
        n = 100
        np.random.seed(42)

        start_date = date(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n)]

        # Clean factor returns
        factor = 0.01 * np.random.randn(n)

        # Asset returns with true beta = 0.8
        asset = 0.001 + 0.8 * factor + 0.005 * np.random.randn(n)

        # Add outliers (5% of data)
        outlier_idx = np.random.choice(n, size=5, replace=False)
        asset[outlier_idx] += 0.1  # Large positive shocks

        data = pl.DataFrame({
            "date": dates,
            "Asset": asset,
            "Factor": factor
        })

        # Fit LS (affected by outliers)
        model_ls = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor"],
            fit_method="LS"
        )

        # Fit Robust (resistant to outliers)
        model_robust = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor"],
            fit_method="Robust",
            family="bisquare"
        )

        # Robust beta should be closer to true value (0.8)
        # LS beta will be biased upward by outliers
        assert abs(model_robust.beta[0, 0] - 0.8) < abs(model_ls.beta[0, 0] - 0.8)

        # Note: Robust resid_sd uses different scale estimation (MAD-based)
        # so direct comparison with LS sigma is not meaningful


class TestRobustUtilities:
    """Test robust utility functions."""

    def test_compute_robust_scale_mad(self):
        """Test MAD scale estimation."""
        # Standard normal data: MAD should be close to 1.0
        np.random.seed(42)
        data = np.random.randn(1000)

        mad = compute_robust_scale(data, method="mad")

        # MAD of standard normal is ~1.0 (with 1.4826 constant)
        assert 0.9 < mad < 1.1

    def test_compute_robust_scale_with_outliers(self):
        """Verify MAD is robust to outliers."""
        # Data with outliers
        data = np.concatenate([
            np.random.randn(950),  # 95% clean data
            np.random.randn(50) * 10  # 5% outliers
        ])

        mad = compute_robust_scale(data, method="mad")

        # MAD should still be close to 1.0 (not inflated by outliers)
        assert 0.8 < mad < 1.3

        # Compare with standard deviation (inflated by outliers)
        std = np.std(data)
        assert mad < std * 0.5  # MAD much smaller than SD

    def test_robust_r_squared(self):
        """Test robust R-squared computation."""
        # Perfect fit
        y = np.array([1, 2, 3, 4, 5])
        fitted = y.copy()
        residuals = y - fitted

        r2 = robust_r_squared(y, fitted, residuals)
        assert r2 > 0.99

        # Poor fit
        fitted_poor = np.mean(y) * np.ones_like(y)
        residuals_poor = y - fitted_poor

        r2_poor = robust_r_squared(y, fitted_poor, residuals_poor)
        assert r2_poor < 0.5


class TestRobustCustomParameters:
    """Test robust regression with custom parameters."""

    def test_custom_tuning_constant(self, synthetic_returns_simple):
        """Test robust regression with custom tuning constant."""
        # Bisquare with custom tuning
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            fit_method="Robust",
            family="bisquare",
            tuning_psi=5.0  # Higher = more robust but less efficient
        )

        assert model.r_squared[0] > 0

    def test_custom_max_iterations(self, synthetic_returns_simple):
        """Test robust regression with custom max iterations."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            fit_method="Robust",
            max_it=50  # Custom max iterations
        )

        assert model.r_squared[0] > 0

    def test_custom_tolerance(self, synthetic_returns_simple):
        """Test robust regression with custom convergence tolerance."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            fit_method="Robust",
            rel_tol=1e-6  # Stricter convergence
        )

        assert model.r_squared[0] > 0


class TestRobustValidation:
    """Test robust regression validation and error handling."""

    def test_invalid_family(self, synthetic_returns_simple):
        """Test that invalid robust family raises error."""
        with pytest.raises(ValueError, match="Invalid robust family"):
            fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["Asset1"],
                factor_names=["Factor1"],
                fit_method="Robust",
                family="invalid_family"
            )

    def test_fit_rlm_direct(self):
        """Test direct use of fit_rlm function."""
        # Create simple data
        np.random.seed(42)
        X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
        y = X @ [1, 2, -1] + 0.1 * np.random.randn(100)

        # Fit RLM
        result = fit_rlm(y, X, family="bisquare")

        # Check result structure
        assert hasattr(result, 'params')
        assert hasattr(result, 'resid')
        assert len(result.params) == 3

        # Parameters should be close to [1, 2, -1]
        np.testing.assert_allclose(result.params, [1, 2, -1], rtol=0.1, atol=0.1)
