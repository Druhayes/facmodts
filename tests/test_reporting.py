"""
Tests for reporting functionality (summary, print, predict methods).

This module tests summary, print, and predict methods for TsfmModel and PaFm objects.
"""

import numpy as np
import polars as pl
import pandas as pd
import pytest
from datetime import date, timedelta
from io import StringIO
import sys

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


class TestSummaryTsfm:
    """Tests for summary_tsfm function."""

    def test_summary_basic(self):
        """Test basic summary generation."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
            "Factor1": np.random.randn(n) * 0.02,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        summary = summary_tsfm(model)

        # Check structure
        assert "call_info" in summary
        assert "se_type" in summary
        assert "summaries" in summary

        # Should have one asset summary
        assert len(summary["summaries"]) == 1

        asset_summary = summary["summaries"][0]
        assert asset_summary["asset"] == "Asset1"
        assert "coefficients" in asset_summary
        assert "std_errors" in asset_summary
        assert "t_stats" in asset_summary
        assert "p_values" in asset_summary

    def test_summary_multiple_assets(self):
        """Test summary with multiple assets."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * factor1 + np.random.randn(n) * 0.01,
            "Asset2": 1.2 * factor1 + np.random.randn(n) * 0.015,
            "Factor1": factor1,
        })

        model = fit_tsfm(
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        summary = summary_tsfm(model)

        # Should have two asset summaries
        assert len(summary["summaries"]) == 2
        assert summary["summaries"][0]["asset"] == "Asset1"
        assert summary["summaries"][1]["asset"] == "Asset2"

    def test_summary_with_lars(self):
        """Test summary for LARS (no standard errors)."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        factor1 = np.random.randn(n) * 0.02
        factor2 = np.random.randn(n) * 0.015
        factor3 = np.random.randn(n) * 0.01

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.8 * factor1 + 0.3 * factor2 + np.random.randn(n) * 0.005,
            "Factor1": factor1,
            "Factor2": factor2,
            "Factor3": factor3,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1", "Factor2", "Factor3"],
            data=data,
            fit_method="LS",
            variable_selection="lars",
            lars_criterion="bic",
        )

        summary = summary_tsfm(model)

        # LARS should have None for std errors
        asset_summary = summary["summaries"][0]
        assert asset_summary["std_errors"] is None
        assert asset_summary["t_stats"] is None
        assert asset_summary["p_values"] is None

        # But should still have coefficients
        assert asset_summary["coefficients"] is not None

    def test_summary_se_types(self):
        """Test different standard error types."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
            "Factor1": np.random.randn(n) * 0.02,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        # Default
        summary_default = summary_tsfm(model, se_type="default")
        assert summary_default["se_type"] == "default"

        # HC/HAC not currently implemented, but should accept the parameter
        # (implementation would require lmtest equivalent)

    def test_summary_robust_with_hc_error(self):
        """Test that HC/HAC errors raise error for Robust method."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
            "Factor1": np.random.randn(n) * 0.02,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="Robust",
        )

        with pytest.raises(ValueError, match="HC/HAC standard errors"):
            summary_tsfm(model, se_type="HC")


class TestPrintTsfm:
    """Tests for print_tsfm function."""

    def test_print_basic(self, capsys):
        """Test basic model printing."""
        n = 40
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
            "Factor1": np.random.randn(n) * 0.02,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        # Capture printed output
        print_tsfm(model)
        captured = capsys.readouterr()

        # Check that key elements are printed
        assert "Time Series Factor Model" in captured.out
        assert "Fit method: LS" in captured.out
        assert "Dimensions:" in captured.out
        assert "Asset1" in captured.out
        assert "Factor1" in captured.out

    def test_print_multiple_assets(self, capsys):
        """Test printing with multiple assets."""
        n = 50
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

        print_tsfm(model)
        captured = capsys.readouterr()

        # Both assets should appear
        assert "Asset1" in captured.out
        assert "Asset2" in captured.out


class TestPrintSummaryTsfm:
    """Tests for print_summary_tsfm function."""

    def test_print_summary_basic(self, capsys):
        """Test basic summary printing."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
            "Factor1": np.random.randn(n) * 0.02,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        summary = summary_tsfm(model)
        print_summary_tsfm(summary)
        captured = capsys.readouterr()

        # Check key elements
        assert "Time Series Factor Model Summary" in captured.out
        assert "Asset: Asset1" in captured.out
        assert "Coefficient" in captured.out
        assert "Estimate" in captured.out
        assert "R-squared" in captured.out


class TestPredictTsfm:
    """Tests for predict_tsfm function."""

    def test_predict_insample(self):
        """Test in-sample predictions."""
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

        # In-sample predictions
        predictions = predict_tsfm(model)

        # Should return array (single asset)
        assert isinstance(predictions, (np.ndarray, dict))

        if isinstance(predictions, np.ndarray):
            assert predictions.shape == (1, n)

    def test_predict_outsample(self):
        """Test out-of-sample predictions."""
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

        # Out-of-sample data
        new_factors = pl.DataFrame({
            "Factor1": [0.01, 0.02, -0.01, 0.015]
        })

        predictions = predict_tsfm(model, newdata=new_factors)

        # Should return predictions for 4 new periods
        assert predictions.shape == (1, 4)

    def test_predict_pandas_input(self):
        """Test prediction with pandas DataFrame input."""
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

        # Pandas input
        new_factors_pd = pd.DataFrame({
            "Factor1": [0.01, 0.02, -0.01]
        })

        predictions = predict_tsfm(model, newdata=new_factors_pd)

        assert predictions.shape == (1, 3)

    def test_predict_multiple_assets(self):
        """Test predictions with multiple assets."""
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

        new_factors = pl.DataFrame({
            "Factor1": [0.01, 0.02]
        })

        predictions = predict_tsfm(model, newdata=new_factors)

        # 2 assets x 2 periods
        assert predictions.shape == (2, 2)

    def test_predict_missing_factors_error(self):
        """Test error when newdata missing factors."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
            "Factor1": np.random.randn(n) * 0.02,
        })

        model = fit_tsfm(
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            data=data,
            fit_method="LS",
        )

        # Missing Factor1
        bad_data = pl.DataFrame({
            "SomethingElse": [0.01, 0.02]
        })

        with pytest.raises(ValueError, match="missing required factors"):
            predict_tsfm(model, newdata=bad_data)


class TestPrintPafm:
    """Tests for print_pafm function."""

    def test_print_pafm_basic(self, capsys):
        """Test basic PaFm printing."""
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

        print_pafm(pa)
        captured = capsys.readouterr()

        # Check key elements
        assert "Performance Attribution" in captured.out
        assert "Asset1" in captured.out
        assert "Factor1" in captured.out
        assert "Specific" in captured.out


class TestSummaryPafm:
    """Tests for summary_pafm function."""

    def test_summary_pafm_basic(self, capsys):
        """Test basic PaFm summary."""
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

        summary_pafm(pa)
        captured = capsys.readouterr()

        # Check for both mean and std deviation
        assert "Mean of returns attributed to factors" in captured.out
        assert "Standard Deviation of returns attributed to factors" in captured.out
        assert "Asset1" in captured.out
