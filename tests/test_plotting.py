"""
Tests for plotting functionality.

This module tests visualization functions for factor models and performance attribution.
"""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from facmodts import (
    fit_tsfm,
    pa_fm,
    fit_tsfm_up_dn,
    plot_tsfm,
    plot_pafm,
    plot_tsfm_updn,
)


class TestPlotTsfm:
    """Tests for plot_tsfm function."""

    def test_plot_single_actual_fitted(self):
        """Test single asset actual vs fitted plot."""
        n = 60
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

        # Should not raise error
        plot_tsfm(model, which=1, plot_single=True, asset_name="Asset1")
        plt.close('all')

    def test_plot_single_multiple_plots(self):
        """Test single asset with multiple plot types."""
        n = 60
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

        # Test multiple plot types
        for plot_type in [1, 2, 3, 4, 5, 6, 9, 10]:
            plot_tsfm(model, which=plot_type, plot_single=True, asset_name="Asset1")
            plt.close('all')

    def test_plot_single_acf_plots(self):
        """Test ACF/PACF plots."""
        n = 80
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

        # ACF/PACF plots
        plot_tsfm(model, which=7, plot_single=True, asset_name="Asset1")
        plt.close('all')

        plot_tsfm(model, which=8, plot_single=True, asset_name="Asset1")
        plt.close('all')

    def test_plot_single_missing_asset_error(self):
        """Test error when asset_name missing for multiple assets."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        with pytest.raises(ValueError, match="asset_name is required"):
            plot_tsfm(model, which=1, plot_single=True)

    def test_plot_group_alpha(self):
        """Test group plot of alpha coefficients."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.001 + 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 0.002 + 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        plot_tsfm(model, which=1, plot_single=False)
        plt.close('all')

    def test_plot_group_beta(self):
        """Test group plot of beta coefficients."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        factor2 = np.random.randn(n) * 0.015

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + 0.3 * factor2 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 - 0.2 * factor2 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
            "Factor2": factor2,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            data=data,
            fit_method="LS",
        )

        plot_tsfm(model, which=2, plot_single=False)
        plt.close('all')

    def test_plot_group_actual_fitted(self):
        """Test group plot of actual vs fitted."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        plot_tsfm(model, which=3, plot_single=False)
        plt.close('all')

    def test_plot_group_r_squared(self):
        """Test group plot of R-squared."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Asset3": 0.5 * factor1 + np.random.randn(n) * 0.02,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2", "Asset3"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        plot_tsfm(model, which=4, plot_single=False)
        plt.close('all')

    def test_plot_group_residual_sd(self):
        """Test group plot of residual volatility."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        plot_tsfm(model, which=5, plot_single=False)
        plt.close('all')

    def test_plot_group_residual_correlation(self):
        """Test group plot of residual correlation."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Asset3": 0.5 * factor1 + np.random.randn(n) * 0.02,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2", "Asset3"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        plot_tsfm(model, which=6, plot_single=False)
        plt.close('all')

    def test_plot_group_sd_decomp(self):
        """Test group plot of SD decomposition."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        factor2 = np.random.randn(n) * 0.015

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + 0.3 * factor2 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 - 0.2 * factor2 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
            "Factor2": factor2,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1", "Factor2"],
            data=data,
            fit_method="LS",
        )

        plot_tsfm(model, which=7, plot_single=False)
        plt.close('all')

    def test_plot_group_too_few_assets_error(self):
        """Test error with too few assets for group plots."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        with pytest.raises(ValueError, match="At least 2 assets required"):
            plot_tsfm(model, which=1, plot_single=False)


class TestPlotPafm:
    """Tests for plot_pafm function."""

    def test_plot_single_cumulative(self):
        """Test single asset cumulative attributed returns."""
        n = 60
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

        plot_pafm(pa, which=1, plot_single=True, asset_name="Asset1")
        plt.close('all')

    def test_plot_single_time_series(self):
        """Test single asset time series of attributed returns."""
        n = 60
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

        plot_pafm(pa, which=2, plot_single=True, asset_name="Asset1")
        plt.close('all')

    def test_plot_single_waterfall(self):
        """Test single asset waterfall chart."""
        n = 60
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

        plot_pafm(pa, which=3, plot_single=True, asset_name="Asset1")
        plt.close('all')

    def test_plot_group_cumulative(self):
        """Test group cumulative attributed returns."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        pa = pa_fm(model)

        plot_pafm(pa, which=1, plot_single=False)
        plt.close('all')

    def test_plot_group_time_series(self):
        """Test group time series of attributed returns."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        pa = pa_fm(model)

        plot_pafm(pa, which=2, plot_single=False)
        plt.close('all')

    def test_plot_single_missing_asset_error(self):
        """Test error when asset_name missing."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        pa = pa_fm(model)

        with pytest.raises(ValueError, match="asset_name required"):
            plot_pafm(pa, which=1, plot_single=True)


class TestPlotTsfmUpDn:
    """Tests for plot_tsfm_updn function."""

    def test_plot_updn_basic(self):
        """Test basic up/down market plot."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        mkt = np.random.randn(n) * 0.02
        asset1 = 0.001 + 0.8 * mkt + np.random.randn(n) * 0.01

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1,
            "Market": mkt,
        })

        updn_model = fit_tsfm_up_dn(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="LS",
        )

        plot_tsfm_updn(updn_model, asset_name="Asset1")
        plt.close('all')

    def test_plot_updn_with_sfm(self):
        """Test up/down plot with single factor model."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        mkt = np.random.randn(n) * 0.02
        asset1 = 0.001 + 0.8 * mkt + np.random.randn(n) * 0.01

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset1,
            "Market": mkt,
        })

        updn_model = fit_tsfm_up_dn(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="LS",
        )

        plot_tsfm_updn(updn_model, asset_name="Asset1", add_sfm=True)
        plt.close('all')
