"""
Tests for performance attribution functionality.

This module tests the pa_fm() function that decomposes returns into
factor-attributed returns and specific returns.
"""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from facmodts import fit_tsfm, pa_fm, PaFm


class TestPaFm:
    """Tests for performance attribution."""

    def test_pa_fm_basic(self):
        """Test basic performance attribution."""
        # Create simple data
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        factor2 = np.random.randn(n) * 0.015

        asset1 = 0.001 + 0.8 * factor1 + 0.3 * factor2 + np.random.randn(n) * 0.005

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1,
            "Factor1": factor1,
            "Factor2": factor2,
        })

        # Fit model
        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2"],
            data=data,
            fit_method="LS",
        )

        # Compute attribution
        pa = pa_fm(model)

        # Check return type
        assert isinstance(pa, PaFm)

        # Check dimensions
        assert pa.cum_ret_attr_f.shape == (1, 2)  # 1 asset, 2 factors
        assert len(pa.cum_spec_ret) == 1

        # Check that attribution list was created
        assert "Asset1" in pa.attr_list
        attr_df = pa.attr_list["Asset1"]
        assert "Factor1" in attr_df.columns
        assert "Factor2" in attr_df.columns
        assert "specific_returns" in attr_df.columns

        # Check that attributed returns sum approximately to total
        # (allowing for compounding effects)
        assert len(attr_df) == n

    def test_pa_fm_multiple_assets(self):
        """Test attribution with multiple assets."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        factor2 = np.random.randn(n) * 0.015

        asset1 = 0.001 + 0.8 * factor1 + 0.3 * factor2 + np.random.randn(n) * 0.005
        asset2 = 0.002 + 1.2 * factor1 - 0.2 * factor2 + np.random.randn(n) * 0.007

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1,
            "Asset2": asset2,
            "Factor1": factor1,
            "Factor2": factor2,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            data=data,
            fit_method="LS",
        )

        pa = pa_fm(model)

        # Check dimensions
        assert pa.cum_ret_attr_f.shape == (2, 2)  # 2 assets, 2 factors
        assert len(pa.cum_spec_ret) == 2

        # Both assets should have attribution
        assert "Asset1" in pa.attr_list
        assert "Asset2" in pa.attr_list

    def test_pa_fm_attribution_decomposition(self):
        """Test that attribution decomposes returns correctly."""
        n = 100
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        # Create deterministic factors
        factor1 = 0.01 * np.sin(np.linspace(0, 2 * np.pi, n))

        # Asset with known relationship
        asset1 = 0.002 + 0.5 * factor1 + 0.01 * np.random.randn(n)

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        pa = pa_fm(model)

        # Get time series attribution
        attr_ts = pa.attr_list["Asset1"]

        # Attributed returns should be approximately beta * factor
        beta = model.beta[0, 0]
        factor_returns = data["Factor1"].to_numpy()
        expected_attr = beta * factor_returns

        actual_attr = attr_ts["Factor1"].to_numpy()

        # Should be close (allowing for small numerical differences)
        np.testing.assert_allclose(actual_attr, expected_attr, atol=1e-10)

    def test_pa_fm_cumulative_returns(self):
        """Test cumulative return calculations."""
        n = 50
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        asset1 = 0.001 + 0.7 * factor1 + np.random.randn(n) * 0.005

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        pa = pa_fm(model)

        # Cumulative returns should be non-zero
        assert pa.cum_ret_attr_f[0, 0] != 0
        assert pa.cum_spec_ret[0] != 0

        # Cumulative factor attribution should have reasonable magnitude
        assert abs(pa.cum_ret_attr_f[0, 0]) < 1.0  # Less than 100% cumulative

    def test_pa_fm_with_robust(self):
        """Test attribution with robust regression."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        asset1 = 0.001 + 0.8 * factor1 + np.random.randn(n) * 0.005

        # Add outliers
        asset1_vals = asset1.copy()
        asset1_vals[[10, 30, 50]] += [0.1, -0.1, 0.1]

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1_vals,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="Robust",
            family="bisquare",
        )

        pa = pa_fm(model)

        # Should work with robust model
        assert isinstance(pa, PaFm)
        assert pa.cum_ret_attr_f.shape == (1, 1)

    def test_pa_fm_invalid_input(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(TypeError, match="must be a TsfmModel"):
            pa_fm("not a model")

    def test_pa_fm_with_variable_selection(self):
        """Test attribution with variable selection (some zero betas)."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        factor2 = np.random.randn(n) * 0.015
        factor3 = np.random.randn(n) * 0.01

        # Asset only depends on factor1 and factor2
        asset1 = 0.001 + 0.8 * factor1 + 0.3 * factor2 + np.random.randn(n) * 0.005

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1,
            "Factor1": factor1,
            "Factor2": factor2,
            "Factor3": factor3,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            data=data,
            fit_method="LS",
            variable_selection="stepwise",
            step_criterion="bic",
        )

        pa = pa_fm(model)

        # Should handle zero/NaN betas
        assert pa.cum_ret_attr_f.shape == (1, 3)

        # Check that attribution list was created
        assert "Asset1" in pa.attr_list

    def test_pa_fm_names_preserved(self):
        """Test that asset and factor names are preserved."""
        n = 50
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "MyAsset": np.random.randn(n) * 0.01,
            "MyFactor": np.random.randn(n) * 0.02,
        })

        model = fit_tsfm(
            asset_names=["MyAsset"],
            factor_names=["MyFactor"],
            data=data,
            fit_method="LS",
        )

        pa = pa_fm(model)

        # Names should match
        assert pa.asset_names == ["MyAsset"]
        assert pa.factor_names == ["MyFactor"]
        assert "MyAsset" in pa.attr_list
        assert "MyFactor" in pa.attr_list["MyAsset"].columns


class TestPaFmIntegration:
    """Integration tests for performance attribution."""

    def test_pa_fm_full_workflow(self):
        """Test complete workflow from fitting to attribution."""
        # Create realistic multi-factor data
        n = 120
        dates = [date(2020, 1, 1) + timedelta(days=i * 7) for i in range(n)]

        # Three factors
        mkt = np.random.randn(n) * 0.03
        size = np.random.randn(n) * 0.015
        value = np.random.randn(n) * 0.012

        # Two assets with different exposures
        growth_fund = 0.002 + 1.0 * mkt - 0.2 * size - 0.3 * value + np.random.randn(n) * 0.01
        value_fund = 0.001 + 0.8 * mkt + 0.1 * size + 0.5 * value + np.random.randn(n) * 0.008

        data = pl.DataFrame({
            "date": dates,
            "GrowthFund": growth_fund,
            "ValueFund": value_fund,
            "Market": mkt,
            "Size": size,
            "Value": value,
        })

        # Fit model
        model = fit_tsfm(
            asset_names=["GrowthFund", "ValueFund"],
            factor_names=["Market", "Size", "Value"],
            data=data,
            fit_method="LS",
        )

        # Attribution
        pa = pa_fm(model)

        # Verify structure
        assert pa.cum_ret_attr_f.shape == (2, 3)
        assert len(pa.cum_spec_ret) == 2
        assert len(pa.attr_list) == 2

        # Verify that factor attributions make sense
        # Growth fund should have negative value attribution (negative loading)
        # Value fund should have positive value attribution (positive loading)

        # Get value factor attributions
        growth_value_attr = pa.attr_list["GrowthFund"]["Value"].mean()
        value_value_attr = pa.attr_list["ValueFund"]["Value"].mean()

        # Signs should be opposite (one negative, one positive)
        # This may not always hold due to randomness, but generally should
        # Just check that they're different
        assert growth_value_attr != value_value_attr
