"""
Tests for risk decomposition functionality.

Tests cover:
- Factor model covariance matrix
- Standard deviation decomposition
- Value-at-Risk decomposition
- Expected Shortfall decomposition
"""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from facmodts import fit_tsfm, fm_cov, fm_sd_decomp, fm_var_decomp, fm_es_decomp


class TestFmCov:
    """Tests for factor model covariance matrix."""

    def test_fm_cov_basic(self, synthetic_returns_simple):
        """Test basic covariance matrix computation."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        cov_matrix = fm_cov(model)

        # Should be 2x2 matrix
        assert cov_matrix.shape == (2, 2)

        # Should be symmetric
        np.testing.assert_allclose(cov_matrix, cov_matrix.T)

        # Diagonal should be positive (variances)
        assert np.all(np.diag(cov_matrix) > 0)

    def test_fm_cov_single_asset(self, synthetic_returns_simple):
        """Test covariance for single asset."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        cov_matrix = fm_cov(model)

        # Should be 1x1 matrix (scalar variance)
        assert cov_matrix.shape == (1, 1)
        assert cov_matrix[0, 0] > 0

    def test_fm_cov_with_custom_factor_cov(self, synthetic_returns_simple):
        """Test with user-specified factor covariance."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        # Custom factor covariance (2x2)
        custom_factor_cov = np.array([[0.0001, 0.00002], [0.00002, 0.00015]])

        cov_matrix = fm_cov(model, factor_cov=custom_factor_cov)

        assert cov_matrix.shape == (1, 1)
        assert cov_matrix[0, 0] > 0

    def test_fm_cov_positive_semidefinite(self, synthetic_returns_simple):
        """Test that covariance matrix is positive semidefinite."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2", "Asset3"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        cov_matrix = fm_cov(model)

        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors


class TestFmSdDecomp:
    """Tests for standard deviation decomposition."""

    def test_sd_decomp_basic(self, synthetic_returns_simple):
        """Test basic SD decomposition."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_sd_decomp(model)

        # Check return structure
        assert "sd_fm" in decomp
        assert "m_sd" in decomp
        assert "c_sd" in decomp
        assert "pc_sd" in decomp

        # Check shapes (1 asset, 2 factors + 1 residual = 3)
        assert decomp["sd_fm"].shape == (1,)
        assert decomp["m_sd"].shape == (1, 3)
        assert decomp["c_sd"].shape == (1, 3)
        assert decomp["pc_sd"].shape == (1, 3)

        # SD should be positive
        assert decomp["sd_fm"][0] > 0

    def test_sd_decomp_euler_property(self, synthetic_returns_simple):
        """Test Euler decomposition property: SD = sum(cSd)."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_sd_decomp(model)

        # Euler property: SD = sum of component SD
        sd_from_components = np.sum(decomp["c_sd"], axis=1)
        np.testing.assert_allclose(decomp["sd_fm"], sd_from_components, rtol=1e-10)

    def test_sd_decomp_percentage_sums_to_100(self, synthetic_returns_simple):
        """Test that percentage contributions sum to 100%."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_sd_decomp(model)

        # Sum of percentage contributions should be 100 for each asset
        pc_sums = np.sum(decomp["pc_sd"], axis=1)
        np.testing.assert_allclose(pc_sums, [100, 100], rtol=1e-10)

    def test_sd_decomp_multiple_assets(self, synthetic_returns_simple):
        """Test SD decomposition with multiple assets."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2", "Asset3"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_sd_decomp(model)

        # Check shapes (3 assets, 2 factors + 1 residual)
        assert decomp["sd_fm"].shape == (3,)
        assert decomp["m_sd"].shape == (3, 3)
        assert decomp["c_sd"].shape == (3, 3)
        assert decomp["pc_sd"].shape == (3, 3)

        # All SDs should be positive
        assert np.all(decomp["sd_fm"] > 0)

    def test_sd_decomp_with_robust(self, synthetic_returns_simple):
        """Test SD decomposition with robust regression."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="Robust",
            family="bisquare"
        )

        decomp = fm_sd_decomp(model)

        # Should still satisfy Euler property
        sd_from_components = np.sum(decomp["c_sd"], axis=1)
        np.testing.assert_allclose(decomp["sd_fm"], sd_from_components, rtol=1e-10)


class TestFmVarDecomp:
    """Tests for Value-at-Risk decomposition."""

    def test_var_decomp_basic_np(self, synthetic_returns_simple):
        """Test basic non-parametric VaR decomposition."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_var_decomp(model, p=0.05, type="np")

        # Check return structure
        assert "var_fm" in decomp
        assert "n_exceed" in decomp
        assert "idx_exceed" in decomp
        assert "m_var" in decomp
        assert "c_var" in decomp
        assert "pc_var" in decomp

        # Check shapes
        assert decomp["var_fm"].shape == (1,)
        assert decomp["n_exceed"].shape == (1,)
        assert len(decomp["idx_exceed"]) == 1
        assert decomp["m_var"].shape == (1, 3)
        assert decomp["c_var"].shape == (1, 3)
        assert decomp["pc_var"].shape == (1, 3)

        # VaR should be negative (loss)
        assert decomp["var_fm"][0] < 0

        # Number of exceedances should be reasonable (around 5% of 100 obs = 5)
        assert 0 <= decomp["n_exceed"][0] <= 15

    def test_var_decomp_euler_property(self, synthetic_returns_simple):
        """Test Euler property: VaR = sum(cVaR)."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_var_decomp(model, p=0.05, type="np")

        # Euler property: VaR = sum of component VaR
        var_from_components = np.sum(decomp["c_var"], axis=1)
        np.testing.assert_allclose(decomp["var_fm"], var_from_components, rtol=1e-10)

    def test_var_decomp_percentage_sums_to_100(self, synthetic_returns_simple):
        """Test that percentage contributions sum to 100%."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_var_decomp(model, p=0.05, type="np")

        # Sum should be 100% for each asset
        pc_sums = np.sum(decomp["pc_var"], axis=1)
        np.testing.assert_allclose(pc_sums, [100, 100], rtol=1e-10)

    def test_var_decomp_normal(self, synthetic_returns_simple):
        """Test normal (parametric) VaR decomposition."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_var_decomp(model, p=0.05, type="normal")

        # Check basic structure
        assert decomp["var_fm"].shape == (1,)
        assert decomp["m_var"].shape == (1, 3)

        # Euler property should still hold
        var_from_components = np.sum(decomp["c_var"], axis=1)
        np.testing.assert_allclose(decomp["var_fm"], var_from_components, rtol=1e-10)

    def test_var_decomp_different_p_values(self, synthetic_returns_simple):
        """Test VaR decomposition with different tail probabilities."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        # 1% VaR (more extreme)
        decomp_1pct = fm_var_decomp(model, p=0.01, type="np")

        # 5% VaR (less extreme)
        decomp_5pct = fm_var_decomp(model, p=0.05, type="np")

        # 1% VaR should be more negative (larger loss)
        assert decomp_1pct["var_fm"][0] < decomp_5pct["var_fm"][0]

        # Fewer exceedances for 1%
        assert decomp_1pct["n_exceed"][0] < decomp_5pct["n_exceed"][0]


class TestFmEsDecomp:
    """Tests for Expected Shortfall decomposition."""

    def test_es_decomp_basic_np(self, synthetic_returns_simple):
        """Test basic non-parametric ES decomposition."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_es_decomp(model, p=0.05, type="np")

        # Check return structure
        assert "es_fm" in decomp
        assert "m_es" in decomp
        assert "c_es" in decomp
        assert "pc_es" in decomp

        # Check shapes
        assert decomp["es_fm"].shape == (1,)
        assert decomp["m_es"].shape == (1, 3)
        assert decomp["c_es"].shape == (1, 3)
        assert decomp["pc_es"].shape == (1, 3)

        # ES should be negative (average tail loss)
        assert decomp["es_fm"][0] < 0

    def test_es_decomp_euler_property(self, synthetic_returns_simple):
        """Test Euler property: ES = sum(cES)."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_es_decomp(model, p=0.05, type="np")

        # Euler property
        es_from_components = np.sum(decomp["c_es"], axis=1)
        np.testing.assert_allclose(decomp["es_fm"], es_from_components, rtol=1e-10)

    def test_es_decomp_percentage_sums_to_100(self, synthetic_returns_simple):
        """Test that percentage contributions sum to 100%."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_es_decomp(model, p=0.05, type="np")

        # Sum should be 100%
        pc_sums = np.sum(decomp["pc_es"], axis=1)
        np.testing.assert_allclose(pc_sums, [100, 100], rtol=1e-10)

    def test_es_decomp_vs_var(self, synthetic_returns_simple):
        """Test that ES >= VaR (more extreme)."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        var_decomp = fm_var_decomp(model, p=0.05, type="np")
        es_decomp = fm_es_decomp(model, p=0.05, type="np")

        # ES should be more negative than VaR (larger loss in tail)
        assert es_decomp["es_fm"][0] <= var_decomp["var_fm"][0]

    def test_es_decomp_normal(self, synthetic_returns_simple):
        """Test normal (parametric) ES decomposition."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_es_decomp(model, p=0.05, type="normal")

        # Check structure
        assert decomp["es_fm"].shape == (1,)
        assert decomp["m_es"].shape == (1, 3)

        # Euler property
        es_from_components = np.sum(decomp["c_es"], axis=1)
        np.testing.assert_allclose(decomp["es_fm"], es_from_components, rtol=1e-10)

    def test_es_decomp_multiple_assets(self, synthetic_returns_simple):
        """Test ES decomposition with multiple assets."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1", "Asset2", "Asset3"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        decomp = fm_es_decomp(model, p=0.05, type="np")

        # Check shapes (3 assets, 2 factors + 1 residual)
        assert decomp["es_fm"].shape == (3,)
        assert decomp["m_es"].shape == (3, 3)
        assert decomp["c_es"].shape == (3, 3)
        assert decomp["pc_es"].shape == (3, 3)

        # All ES should be negative
        assert np.all(decomp["es_fm"] < 0)


class TestDecompositionConsistency:
    """Tests for consistency across decomposition methods."""

    def test_all_decompositions_consistent(self, synthetic_returns_simple):
        """Test that all decompositions satisfy Euler property."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        # SD decomposition
        sd_decomp = fm_sd_decomp(model)
        sd_reconstructed = np.sum(sd_decomp["c_sd"], axis=1)
        np.testing.assert_allclose(sd_decomp["sd_fm"], sd_reconstructed, rtol=1e-10)

        # VaR decomposition
        var_decomp = fm_var_decomp(model, p=0.05, type="np")
        var_reconstructed = np.sum(var_decomp["c_var"], axis=1)
        np.testing.assert_allclose(var_decomp["var_fm"], var_reconstructed, rtol=1e-10)

        # ES decomposition
        es_decomp = fm_es_decomp(model, p=0.05, type="np")
        es_reconstructed = np.sum(es_decomp["c_es"], axis=1)
        np.testing.assert_allclose(es_decomp["es_fm"], es_reconstructed, rtol=1e-10)

    def test_percentage_contributions_all_sum_to_100(self, synthetic_returns_simple):
        """Test all percentage contributions sum to 100%."""
        model = fit_tsfm(
            data=synthetic_returns_simple,
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            fit_method="LS"
        )

        # All should sum to 100%
        sd_decomp = fm_sd_decomp(model)
        assert np.isclose(np.sum(sd_decomp["pc_sd"]), 100, rtol=1e-10)

        var_decomp = fm_var_decomp(model, p=0.05, type="np")
        assert np.isclose(np.sum(var_decomp["pc_var"]), 100, rtol=1e-10)

        es_decomp = fm_es_decomp(model, p=0.05, type="np")
        assert np.isclose(np.sum(es_decomp["pc_es"]), 100, rtol=1e-10)


class TestDecompositionEdgeCases:
    """Tests for edge cases in decomposition."""

    def test_single_factor_decomposition(self):
        """Test decomposition with single factor."""
        # Create simple data
        n = 100
        np.random.seed(42)

        start_date = date(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n)]

        factor = 0.01 * np.random.randn(n)
        asset = 0.001 + 0.8 * factor + 0.005 * np.random.randn(n)

        data = pl.DataFrame({
            "date": dates,
            "Asset": asset,
            "Factor": factor
        })

        model = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor"],
            fit_method="LS"
        )

        # SD decomposition (1 factor + 1 residual = 2 components)
        sd_decomp = fm_sd_decomp(model)
        assert sd_decomp["c_sd"].shape == (1, 2)

        # Euler property still holds
        assert np.isclose(np.sum(sd_decomp["c_sd"]), sd_decomp["sd_fm"][0])

    def test_perfect_fit_decomposition(self):
        """Test decomposition when model fits perfectly."""
        # Create perfect fit data (no noise)
        n = 100
        np.random.seed(42)

        start_date = date(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n)]

        factor = 0.01 * np.random.randn(n)
        asset = 0.001 + 0.8 * factor  # No noise

        data = pl.DataFrame({
            "date": dates,
            "Asset": asset,
            "Factor": factor
        })

        model = fit_tsfm(
            data=data,
            asset_names=["Asset"],
            factor_names=["Factor"],
            fit_method="LS"
        )

        sd_decomp = fm_sd_decomp(model)

        # Residual contribution should be near zero for perfect fit
        assert sd_decomp["pc_sd"][0, 1] < 1  # Less than 1% from residual
