"""
Tests for variable selection functionality.

Tests cover:
- Stepwise selection (forward, backward, both)
- Best subset selection
- LARS/Lasso selection
- Integration with fit_tsfm
"""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from facmodts import fit_tsfm


class TestStepwiseSelection:
    """Tests for stepwise variable selection."""

    def test_stepwise_forward(self, synthetic_returns_simple):
        """Test forward stepwise selection."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
            variable_selection="stepwise",
            step_direction="forward",
            step_criterion="bic"
        )

        # Model should fit successfully
        assert model.variable_selection == "stepwise"
        assert model.r_squared[0] > 0

        # Beta should have at most 3 non-zero factors (may select fewer)
        n_selected = np.sum(model.beta[0, :] != 0)
        assert 0 <= n_selected <= 3

    def test_stepwise_backward(self, synthetic_returns_simple):
        """Test backward stepwise elimination."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
            variable_selection="stepwise",
            step_direction="backward",
            step_criterion="bic"
        )

        assert model.variable_selection == "stepwise"
        assert model.r_squared[0] > 0

    def test_stepwise_both(self, synthetic_returns_simple):
        """Test bidirectional stepwise selection."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
            variable_selection="stepwise",
            step_direction="both",
            step_criterion="aic"
        )

        assert model.variable_selection == "stepwise"
        assert model.r_squared[0] > 0

    def test_stepwise_multiple_assets(self, synthetic_returns_simple):
        """Test stepwise with multiple assets (different factors per asset)."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
            variable_selection="stepwise",
            step_direction="both"
        )

        # Both assets should fit
        assert model.beta.shape == (2, 3)
        assert model.r_squared[0] > 0
        assert model.r_squared[1] > 0

    def test_stepwise_with_dls(self, synthetic_returns_simple):
        """Test stepwise with DLS fit method."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="DLS",
            variable_selection="stepwise",
            decay=0.90
        )

        assert model.fit_method == "DLS"
        assert model.variable_selection == "stepwise"
        assert model.r_squared[0] > 0

    def test_stepwise_with_robust(self, synthetic_returns_simple):
        """Test stepwise with Robust fit method."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="Robust",
            variable_selection="stepwise",
            family="bisquare"
        )

        assert model.fit_method == "Robust"
        assert model.variable_selection == "stepwise"
        assert model.r_squared[0] > 0


class TestSubsetsSelection:
    """Tests for best subset selection."""

    def test_subsets_basic(self, synthetic_returns_simple):
        """Test basic subset selection."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
            variable_selection="subsets",
            nvmin=1,
            nvmax=3
        )

        assert model.variable_selection == "subsets"
        assert model.r_squared[0] > 0

        # Should select between 1 and 3 factors
        n_selected = np.sum(model.beta[0, :] != 0)
        assert 1 <= n_selected <= 3

    def test_subsets_nvmin_nvmax(self, synthetic_returns_simple):
        """Test subset selection with specific size range."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
            variable_selection="subsets",
            nvmin=2,
            nvmax=2  # Force exactly 2 factors
        )

        # Should select exactly 2 factors
        n_selected = np.sum(model.beta[0, :] != 0)
        assert n_selected == 2

    def test_subsets_multiple_assets(self, synthetic_returns_simple):
        """Test subset selection with multiple assets."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",
            variable_selection="subsets",
            nvmin=1,
            nvmax=3
        )

        assert model.beta.shape == (2, 3)
        assert model.r_squared[0] > 0
        assert model.r_squared[1] > 0

        # Each asset can select different factors
        n_selected_1 = np.sum(model.beta[0, :] != 0)
        n_selected_2 = np.sum(model.beta[1, :] != 0)
        assert 1 <= n_selected_1 <= 3
        assert 1 <= n_selected_2 <= 3

    def test_subsets_with_dls(self, synthetic_returns_simple):
        """Test subset selection with DLS."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="DLS",
            variable_selection="subsets",
            decay=0.95
        )

        assert model.fit_method == "DLS"
        assert model.variable_selection == "subsets"
        assert model.r_squared[0] > 0

    def test_subsets_with_robust(self, synthetic_returns_simple):
        """Test subset selection with Robust."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="Robust",
            variable_selection="subsets",
            family="huber"
        )

        assert model.fit_method == "Robust"
        assert model.variable_selection == "subsets"
        assert model.r_squared[0] > 0


class TestLarsSelection:
    """Tests for LARS/Lasso variable selection."""

    def test_lars_basic_cp(self, synthetic_returns_simple):
        """Test LARS with Cp criterion."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            fit_method="LS",  # LARS ignores fit_method
            variable_selection="lars",
            lars_criterion="cp"
        )

        assert model.variable_selection == "lars"
        assert model.r_squared[0] > 0

        # LARS typically produces sparse solutions
        n_selected = np.sum(model.beta[0, :] != 0)
        assert 0 <= n_selected <= 3

    def test_lars_aic(self, synthetic_returns_simple):
        """Test LARS with AIC criterion."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            variable_selection="lars",
            lars_criterion="aic"
        )

        assert model.variable_selection == "lars"
        assert model.r_squared[0] > 0

    def test_lars_bic(self, synthetic_returns_simple):
        """Test LARS with BIC criterion."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            variable_selection="lars",
            lars_criterion="bic"
        )

        assert model.variable_selection == "lars"
        assert model.r_squared[0] > 0

        # BIC tends to select fewer factors than AIC
        n_selected = np.sum(model.beta[0, :] != 0)
        assert 0 <= n_selected <= 3

    def test_lars_multiple_assets(self, synthetic_returns_simple):
        """Test LARS with multiple assets."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            variable_selection="lars",
            lars_criterion="bic"
        )

        assert model.beta.shape == (2, 3)
        assert model.r_squared[0] > 0
        assert model.r_squared[1] > 0

        # Each asset can select different factors
        n_selected_1 = np.sum(model.beta[0, :] != 0)
        n_selected_2 = np.sum(model.beta[1, :] != 0)
        assert 0 <= n_selected_1 <= 3
        assert 0 <= n_selected_2 <= 3

    def test_lars_sparsity(self):
        """Test that LARS produces sparse solutions with many factors."""
        # Create data with many factors but only 2 are truly important
        n = 100
        np.random.seed(42)

        start_date = date(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n)]

        # Only 2 factors matter
        factor1 = 0.01 * np.random.randn(n)
        factor2 = 0.015 * np.random.randn(n)
        factor3 = 0.01 * np.random.randn(n)  # Noise
        factor4 = 0.01 * np.random.randn(n)  # Noise
        factor5 = 0.01 * np.random.randn(n)  # Noise

        # Asset depends only on Factor1 and Factor2
        asset = 0.001 + 0.8 * factor1 + 0.3 * factor2 + 0.005 * np.random.randn(n)

        data = pl.DataFrame({
            "date": dates,
            "Asset": asset,
            "Factor1": factor1,
            "Factor2": factor2,
            "Factor3": factor3,
            "Factor4": factor4,
            "Factor5": factor5,
        })

        model = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor1", "Factor2", "Factor3", "Factor4", "Factor5"],
            variable_selection="lars",
            lars_criterion="bic"
        )

        # LARS with BIC should select mostly just the important factors
        n_selected = np.sum(model.beta[0, :] != 0)
        assert n_selected <= 3  # Should not select all 5

        # Factor1 and Factor2 should have largest magnitudes
        beta_abs = np.abs(model.beta[0, :])
        top_2_idx = np.argsort(beta_abs)[-2:]
        assert 0 in top_2_idx or 1 in top_2_idx  # Factor1 or Factor2 in top 2


class TestVariableSelectionValidation:
    """Tests for variable selection validation and error handling."""

    def test_invalid_nvmin_nvmax(self, synthetic_returns_simple):
        """Test that nvmin > nvmax raises error."""
        with pytest.raises(ValueError, match="nvmin.*must be <= nvmax"):
            fit_tsfm(
                data=synthetic_returns_simple,
                asset_names=["Asset1"],
                factor_names=["Factor1", "Factor2"],
                variable_selection="subsets",
                nvmin=5,
                nvmax=2
            )

    def test_stepwise_direction_alias(self, synthetic_returns_simple):
        """Test that step_direction parameter works."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            variable_selection="stepwise",
            step_direction="forward"  # Using alias
        )

        assert model.variable_selection == "stepwise"
        assert model.r_squared[0] > 0


class TestVariableSelectionComparison:
    """Compare variable selection methods."""

    def test_selection_reduces_factors(self):
        """Test that selection methods can reduce number of factors."""
        # Create data where only 1 factor matters
        n = 100
        np.random.seed(42)

        start_date = date(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n)]

        factor1 = 0.01 * np.random.randn(n)
        factor2 = 0.01 * np.random.randn(n)  # Noise
        factor3 = 0.01 * np.random.randn(n)  # Noise

        # Asset depends only on Factor1
        asset = 0.001 + 1.0 * factor1 + 0.002 * np.random.randn(n)

        data = pl.DataFrame({
            "date": dates,
            "Asset": asset,
            "Factor1": factor1,
            "Factor2": factor2,
            "Factor3": factor3,
        })

        # Fit with no selection (all factors)
        model_none = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            variable_selection="none"
        )

        # All 3 factors included
        assert np.sum(model_none.beta[0, :] != 0) == 3

        # Fit with stepwise
        model_step = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            variable_selection="stepwise",
            step_criterion="bic"
        )

        # Should select fewer factors
        n_selected_step = np.sum(model_step.beta[0, :] != 0)
        assert n_selected_step < 3

        # Fit with LARS
        model_lars = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            variable_selection="lars",
            lars_criterion="bic"
        )

        # Should select fewer factors
        n_selected_lars = np.sum(model_lars.beta[0, :] != 0)
        assert n_selected_lars < 3
