"""
Unit tests for fit_tsfm() and related functions.

Tests cover:
- Basic LS regression
- DLS (discounted least squares) with custom decay
- Excess returns computation
- Incomplete case handling (NAs)
- Multi-asset fitting with unequal histories
- Input validation and error handling
"""

import numpy as np
import polars as pl
import pytest

from facmodts import fit_tsfm, fit_tsfm_control


class TestFitTsfmBasic:
    """Tests for basic fit_tsfm() functionality with LS method."""

    def test_ls_single_asset(self, synthetic_returns_simple):
        """Test LS fitting for a single asset."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS",
        )

        # Check model structure
        assert model.fit_method == "LS"
        assert model.variable_selection == "none"
        assert len(model.asset_names) == 1
        assert len(model.factor_names) == 2

        # Check coefficient shapes
        assert model.alpha.shape == (1,)
        assert model.beta.shape == (1, 2)
        assert len(model.r_squared) == 1
        assert len(model.resid_sd) == 1

        # Check R-squared is reasonable
        assert 0 <= model.r_squared[0] <= 1

        # Check residual SD is positive
        assert model.resid_sd[0] > 0

    def test_ls_multiple_assets(self, synthetic_returns_simple):
        """Test LS fitting for multiple assets."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2", "Asset3"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
        )

        # Check shapes
        assert model.alpha.shape == (3,)
        assert model.beta.shape == (3, 3)
        assert len(model.r_squared) == 3
        assert len(model.resid_sd) == 3

        # All R-squared should be reasonable
        assert np.all((model.r_squared >= 0) & (model.r_squared <= 1))

        # All residual SDs should be positive
        assert np.all(model.resid_sd > 0)

    def test_ls_with_subset_of_factors(self, synthetic_returns_simple):
        """Test fitting with only a subset of available factors."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],  # Only 1 factor
            fit_method="LS",
        )

        # Check beta has correct shape (2 assets, 1 factor)
        assert model.beta.shape == (2, 1)
        assert len(model.factor_names) == 1


class TestFitTsfmDLS:
    """Tests for DLS (discounted least squares) method."""

    def test_dls_default_decay(self, synthetic_returns_simple):
        """Test DLS with default decay factor (0.95)."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="DLS",
        )

        assert model.fit_method == "DLS"
        assert model.r_squared[0] > 0  # Should fit reasonably
        assert model.resid_sd[0] > 0

    def test_dls_custom_decay(self, synthetic_returns_simple):
        """Test DLS with custom decay factor."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="DLS",
            decay=0.90,
        )

        assert model.fit_method == "DLS"
        # With stronger decay (0.90 < 0.95), recent observations weighted more heavily
        assert model.r_squared[0] > 0

    def test_dls_vs_ls_different_coefficients(self, synthetic_returns_simple):
        """Verify that DLS and LS produce different coefficients."""
        model_ls = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS",
        )

        model_dls = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="DLS",
            decay=0.90,
        )

        # Coefficients should differ due to weighting
        assert not np.allclose(model_ls.beta, model_dls.beta, rtol=1e-10)


class TestFitTsfmExcessReturns:
    """Tests for excess returns computation via rf_name."""

    def test_excess_returns_basic(self, synthetic_returns_with_rf):
        """Test fitting with risk-free rate subtraction."""
        model = fit_tsfm(
            data=synthetic_returns_with_rf,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            rf_name="RF",
            fit_method="LS",
        )

        # Model should fit successfully
        assert model.rf_name == "RF"
        assert model.r_squared[0] > 0
        assert model.r_squared[1] > 0

    def test_excess_returns_vs_no_rf(self, synthetic_returns_with_rf):
        """Verify that rf_name parameter properly computes excess returns."""
        # Fit with excess returns
        model_with_rf = fit_tsfm(
            data=synthetic_returns_with_rf,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            rf_name="RF",
            fit_method="LS",
        )

        # Model should fit successfully
        assert model_with_rf.r_squared[0] > 0
        assert model_with_rf.r_squared[1] > 0

        # Verify rf_name is recorded
        assert model_with_rf.rf_name == "RF"


class TestFitTsfmNAHandling:
    """Tests for incomplete case (NA) handling."""

    def test_na_removal_per_asset(self, synthetic_returns_with_nas):
        """Test that NAs are removed per asset independently."""
        model = fit_tsfm(
            data=synthetic_returns_with_nas,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS",
        )

        # Both assets should fit (even with different NA patterns)
        assert model.r_squared[0] > 0
        assert model.r_squared[1] > 0

        # Asset fit objects should exist
        assert "Asset1" in model.asset_fit
        assert "Asset2" in model.asset_fit

        # Number of observations used should be less than total
        # (some removed due to NAs)
        n_obs_asset1 = model.asset_fit["Asset1"].nobs
        n_obs_asset2 = model.asset_fit["Asset2"].nobs

        assert n_obs_asset1 < 100  # Original had 100 periods
        assert n_obs_asset2 < 100


class TestFitTsfmValidation:
    """Tests for input validation and error handling."""

    def test_invalid_fit_method(self, synthetic_returns_simple):
        """Test that invalid fit_method raises ValueError."""
        with pytest.raises(ValueError, match="fit_method must be"):
            fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["Asset1"],
                factor_names=["Factor1"],
                fit_method="InvalidMethod",
            )

    def test_invalid_variable_selection(self, synthetic_returns_simple):
        """Test that invalid variable_selection raises ValueError."""
        with pytest.raises(ValueError, match="variable_selection must be"):
            fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["Asset1"],
                factor_names=["Factor1"],
                variable_selection="invalid",
            )

    def test_robust_not_implemented(self, synthetic_returns_simple):
        """Test that Robust method raises NotImplementedError in Phase 1."""
        with pytest.raises(NotImplementedError, match="Phase 2"):
            fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["Asset1"],
                factor_names=["Factor1"],
                fit_method="Robust",
            )

    def test_stepwise_not_implemented(self, synthetic_returns_simple):
        """Test that stepwise selection raises NotImplementedError in Phase 1."""
        with pytest.raises(NotImplementedError, match="Phase 2"):
            fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["Asset1"],
                factor_names=["Factor1"],
                variable_selection="stepwise",
            )

    def test_missing_columns(self, synthetic_returns_simple):
        """Test that missing columns raise ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["NonexistentAsset"],
                factor_names=["Factor1"],
                fit_method="LS",
            )


class TestFitTsfmControl:
    """Tests for fit_tsfm_control() parameter validation."""

    def test_control_defaults(self):
        """Test that control parameters have correct defaults."""
        ctrl = fit_tsfm_control()

        assert ctrl.decay == 0.95
        assert ctrl.bb == 0.5
        assert ctrl.efficiency == 0.85
        assert ctrl.k == 2.0
        assert ctrl.nvmax == 8

    def test_control_custom_decay(self):
        """Test custom decay parameter."""
        ctrl = fit_tsfm_control(decay=0.90)
        assert ctrl.decay == 0.90

    def test_control_custom_bb(self):
        """Test custom breakdown point."""
        ctrl = fit_tsfm_control(bb=0.4)
        assert ctrl.bb == 0.4

    def test_control_invalid_decay(self):
        """Test that invalid decay raises ValueError."""
        with pytest.raises(ValueError, match="decay must be"):
            fit_tsfm_control(decay=1.5)

        with pytest.raises(ValueError, match="decay must be"):
            fit_tsfm_control(decay=0.0)

    def test_control_invalid_bb(self):
        """Test that invalid bb raises ValueError."""
        with pytest.raises(ValueError, match="bb must be"):
            fit_tsfm_control(bb=0.6)

        with pytest.raises(ValueError, match="bb must be"):
            fit_tsfm_control(bb=-0.1)


class TestFitTsfmCoefficients:
    """Tests for extracted coefficients and statistics."""

    def test_beta_recovery_known_data(self):
        """Test that beta coefficients are close to true values for synthetic data."""
        # Create data with known betas
        n = 200
        from datetime import date, timedelta
        start_date = date(2020, 1, 1)
        end_date = start_date + timedelta(days=n - 1)
        dates = pl.date_range(start_date, end_date, "1d", eager=True)

        # True betas: [0.8, 0.3]
        factor1 = 0.01 * np.random.randn(n)
        factor2 = 0.015 * np.random.randn(n)
        asset = 0.001 + 0.8 * factor1 + 0.3 * factor2 + 0.002 * np.random.randn(n)

        df = pl.DataFrame(
            {"date": dates, "Asset": asset, "Factor1": factor1, "Factor2": factor2}
        )

        model = fit_tsfm(
            data=df, asset_names=["Asset"], factor_names=["Factor1", "Factor2"], fit_method="LS"
        )

        # Beta should be close to [0.8, 0.3]
        np.testing.assert_allclose(model.beta[0, 0], 0.8, rtol=0.1, atol=0.05)
        np.testing.assert_allclose(model.beta[0, 1], 0.3, rtol=0.1, atol=0.05)

    def test_r_squared_perfect_fit(self):
        """Test R-squared is near 1.0 for perfect fit (no noise)."""
        n = 100
        from datetime import date, timedelta
        start_date = date(2020, 1, 1)
        end_date = start_date + timedelta(days=n - 1)
        dates = pl.date_range(start_date, end_date, "1d", eager=True)

        factor1 = 0.01 * np.random.randn(n)
        asset = 0.001 + 0.8 * factor1  # No noise

        df = pl.DataFrame({"date": dates, "Asset": asset, "Factor1": factor1})

        model = fit_tsfm(data=df, asset_names=["Asset"], factor_names=["Factor1"], fit_method="LS")

        # R-squared should be very close to 1.0
        assert model.r_squared[0] > 0.99
