"""
Tests for convenience wrapper functions.

This module tests the specialized time series factor model wrappers:
- fit_tsfm_mt: Market-timing models
- fit_tsfm_up_dn: Up/down market models
- fit_ff3_model: Fama-French 3-factor
- fit_ff4_model: Fama-French 4-factor
"""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from facmodts import (
    fit_tsfm_mt,
    fit_tsfm_up_dn,
    fit_ff3_model,
    fit_ff4_model,
    TsfmModel,
    TsfmUpDn,
)


class TestFitTsfmMT:
    """Tests for market-timing model wrapper."""

    def test_mt_basic(self):
        """Test basic market-timing model fitting."""
        # Create data with Market column
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        mkt = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * mkt + np.random.randn(n) * 0.01,
            "Market": mkt,
        })

        # Fit market-timing model
        model = fit_tsfm_mt(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="LS",
        )

        # Check return type
        assert isinstance(model, TsfmModel)

        # Should have 2 factors: Market and down_market
        assert len(model.factor_names) == 2
        assert "Market" in model.factor_names
        assert "down_market" in model.factor_names

        # Check beta shape
        assert model.beta.shape == (1, 2)

        # Check that model was fitted
        assert model.r_squared[0] >= 0
        assert model.r_squared[0] <= 1

    def test_mt_with_rf(self):
        """Test market-timing model with risk-free rate."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        mkt = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * mkt + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * mkt + np.random.randn(n) * 0.015,
            "Market": mkt,
            "RF": np.ones(n) * 0.002,
        })

        model = fit_tsfm_mt(
            asset_names=["Asset1", "Asset2"],
            mkt_name="Market",
            rf_name="RF",
            data=data,
            fit_method="LS",
        )

        # Check dimensions
        assert model.beta.shape == (2, 2)
        assert len(model.asset_names) == 2

    def test_mt_down_market_construction(self):
        """Test that down_market factor is correctly constructed."""
        # Create data with known market returns
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        # Market with clear up/down periods
        mkt = np.array([0.02, -0.03, 0.01, -0.02, 0.03, -0.01] * 10)

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.5 * mkt + np.random.randn(n) * 0.01,
            "Market": mkt,
        })

        model = fit_tsfm_mt(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="LS",
        )

        # Check that down_market column was created in data
        assert "down_market" in model.data.columns

        # Verify down_market construction: should be max(0, -R_m)
        down_mkt = model.data["down_market"].to_numpy()
        mkt_vals = model.data["Market"].to_numpy()

        # Where market is positive, down_market should be 0
        assert np.all(down_mkt[mkt_vals > 0] == 0)

        # Where market is negative, down_market should be -mkt_vals
        assert np.allclose(down_mkt[mkt_vals < 0], -mkt_vals[mkt_vals < 0], atol=1e-10)

    def test_mt_missing_mkt_name(self):
        """Test that missing mkt_name raises error."""
        n = 20
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
        })

        with pytest.raises(ValueError, match="mkt_name is required"):
            fit_tsfm_mt(
                asset_names=["Asset1"],
                mkt_name=None,
                data=data,
            )

    def test_mt_robust_fit(self):
        """Test market-timing with robust regression."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        mkt = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * mkt + np.random.randn(n) * 0.01,
            "Market": mkt,
        })

        model = fit_tsfm_mt(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="Robust",
            family="bisquare",
        )

        assert model.fit_method == "Robust"
        assert model.beta.shape == (1, 2)


class TestFitTsfmUpDn:
    """Tests for up/down market model wrapper."""

    def test_updn_basic(self):
        """Test basic up/down market model."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        mkt = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * mkt + np.random.randn(n) * 0.01,
            "Asset2": 1.0 * mkt + np.random.randn(n) * 0.015,
            "Market": mkt,
        })

        result = fit_tsfm_up_dn(
            asset_names=["Asset1", "Asset2"],
            mkt_name="Market",
            data=data,
            fit_method="LS",
        )

        # Check return type
        assert isinstance(result, TsfmUpDn)

        # Check that both models exist
        assert isinstance(result.up_model, TsfmModel)
        assert isinstance(result.down_model, TsfmModel)

        # Check market_name
        assert result.market_name == "Market"

        # Check up_market_flag is boolean
        assert result.up_market_flag.dtype == bool

    def test_updn_split_consistency(self):
        """Test that up/down split is consistent."""
        # Create data with known market returns
        n = 100
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        # 60 up, 40 down
        mkt = np.concatenate([
            np.abs(np.random.randn(60)) * 0.01,  # Up market
            -np.abs(np.random.randn(40)) * 0.01   # Down market
        ])
        np.random.shuffle(mkt)  # Shuffle

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * mkt + np.random.randn(n) * 0.01,
            "Market": mkt,
        })

        result = fit_tsfm_up_dn(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="LS",
        )

        # Count up/down periods
        n_up = np.sum(result.up_market_flag)
        n_down = np.sum(~result.up_market_flag)

        # Check counts match data
        mkt_vals = data["Market"].to_numpy()
        expected_n_up = np.sum(mkt_vals >= 0)
        expected_n_down = np.sum(mkt_vals < 0)

        assert n_up == expected_n_up
        assert n_down == expected_n_down

        # Check that models have correct number of observations
        # Up model should have n_up periods
        assert result.up_model.residuals.shape[1] == n_up
        # Down model should have n_down periods
        assert result.down_model.residuals.shape[1] == n_down

    def test_updn_different_betas(self):
        """Test that up/down markets can have different betas."""
        n = 120
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        # Create market returns
        mkt = np.random.randn(n) * 0.02

        # Asset with different sensitivity in up vs down markets
        # High beta (1.5) in up markets, low beta (0.3) in down markets
        asset = np.where(
            mkt >= 0,
            1.5 * mkt + np.random.randn(n) * 0.01,  # Up
            0.3 * mkt + np.random.randn(n) * 0.01   # Down
        )

        data = pl.DataFrame({
            "date": dates,
            "Asset1": asset,
            "Market": mkt,
        })

        result = fit_tsfm_up_dn(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="LS",
        )

        # Up market beta should be higher than down market beta
        # (Since up beta = 1.5, down beta = 0.3)
        beta_up = result.up_model.beta[0, 0]
        beta_down = result.down_model.beta[0, 0]

        # With enough data, this should hold
        assert beta_up > beta_down

    def test_updn_with_rf(self):
        """Test up/down model with risk-free rate."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        mkt = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.8 * mkt + np.random.randn(n) * 0.01,
            "Market": mkt,
            "RF": np.ones(n) * 0.001,
        })

        result = fit_tsfm_up_dn(
            asset_names=["Asset1"],
            mkt_name="Market",
            rf_name="RF",
            data=data,
            fit_method="LS",
        )

        # Models should be fitted
        assert result.up_model.r_squared[0] >= 0
        assert result.down_model.r_squared[0] >= 0

    def test_updn_missing_mkt_name(self):
        """Test that missing mkt_name raises error."""
        n = 20
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        data = pl.DataFrame({
            "date": dates,
            "Asset1": np.random.randn(n) * 0.01,
        })

        with pytest.raises(ValueError, match="mkt_name is required"):
            fit_tsfm_up_dn(
                asset_names=["Asset1"],
                mkt_name=None,
                data=data,
            )

    def test_updn_robust(self):
        """Test up/down with robust regression."""
        n = 100
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        mkt = np.random.randn(n) * 0.02

        data = pl.DataFrame({
            "date": dates,
            "Asset1": 0.6 * mkt + np.random.randn(n) * 0.01,
            "Market": mkt,
        })

        result = fit_tsfm_up_dn(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data,
            fit_method="Robust",
            family="bisquare",
        )

        assert result.up_model.fit_method == "Robust"
        assert result.down_model.fit_method == "Robust"


class TestFitFF3Model:
    """Tests for Fama-French 3-factor model."""

    def test_ff3_basic(self):
        """Test basic FF3 model fitting."""
        # Create sample FF3 data
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "MKT": np.random.randn(n) * 0.02,
            "SMB": np.random.randn(n) * 0.01,
            "HML": np.random.randn(n) * 0.01,
            "Fund": np.random.randn(n) * 0.025,
        })

        model = fit_ff3_model(
            asset_names=["Fund"],
            factor_data=data,
            fit_method="LS",
        )

        # Check return type
        assert isinstance(model, TsfmModel)

        # Should have 3 factors
        assert len(model.factor_names) == 3
        assert "MKT" in model.factor_names
        assert "SMB" in model.factor_names
        assert "HML" in model.factor_names

        # Check beta shape: 1 asset x 3 factors
        assert model.beta.shape == (1, 3)

    def test_ff3_multiple_assets(self):
        """Test FF3 with multiple assets."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        mkt = np.random.randn(n) * 0.02
        smb = np.random.randn(n) * 0.01
        hml = np.random.randn(n) * 0.01

        data = pl.DataFrame({
            "date": dates,
            "MKT": mkt,
            "SMB": smb,
            "HML": hml,
            "Fund1": 0.8 * mkt + 0.3 * smb + 0.1 * hml + np.random.randn(n) * 0.01,
            "Fund2": 1.2 * mkt - 0.2 * smb + 0.5 * hml + np.random.randn(n) * 0.015,
        })

        model = fit_ff3_model(
            asset_names=["Fund1", "Fund2"],
            factor_data=data,
            fit_method="LS",
        )

        assert model.beta.shape == (2, 3)
        assert len(model.asset_names) == 2

    def test_ff3_missing_factors(self):
        """Test error when FF3 factors are missing."""
        data = pl.DataFrame({
            "date": [date(2020, 1, 1)],
            "MKT": [0.01],
            "SMB": [0.005],
            # Missing HML
            "Fund": [0.015],
        })

        with pytest.raises(ValueError, match="missing required columns"):
            fit_ff3_model(
                asset_names=["Fund"],
                factor_data=data,
            )

    def test_ff3_robust(self):
        """Test FF3 with robust regression."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "MKT": np.random.randn(n) * 0.02,
            "SMB": np.random.randn(n) * 0.01,
            "HML": np.random.randn(n) * 0.01,
            "Fund": np.random.randn(n) * 0.025,
        })

        model = fit_ff3_model(
            asset_names=["Fund"],
            factor_data=data,
            fit_method="Robust",
            family="bisquare",
        )

        assert model.fit_method == "Robust"
        assert model.beta.shape == (1, 3)


class TestFitFF4Model:
    """Tests for Fama-French 4-factor (Carhart) model."""

    def test_ff4_basic_mom(self):
        """Test basic FF4 model with MOM factor."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "MKT": np.random.randn(n) * 0.02,
            "SMB": np.random.randn(n) * 0.01,
            "HML": np.random.randn(n) * 0.01,
            "MOM": np.random.randn(n) * 0.015,
            "Fund": np.random.randn(n) * 0.025,
        })

        model = fit_ff4_model(
            asset_names=["Fund"],
            factor_data=data,
            fit_method="LS",
        )

        # Check return type
        assert isinstance(model, TsfmModel)

        # Should have 4 factors
        assert len(model.factor_names) == 4
        assert "MKT" in model.factor_names
        assert "SMB" in model.factor_names
        assert "HML" in model.factor_names
        assert "MOM" in model.factor_names

        # Check beta shape: 1 asset x 4 factors
        assert model.beta.shape == (1, 4)

    def test_ff4_basic_umd(self):
        """Test FF4 model with UMD (up-minus-down) factor."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "MKT": np.random.randn(n) * 0.02,
            "SMB": np.random.randn(n) * 0.01,
            "HML": np.random.randn(n) * 0.01,
            "UMD": np.random.randn(n) * 0.015,  # Alternative name for momentum
            "Fund": np.random.randn(n) * 0.025,
        })

        model = fit_ff4_model(
            asset_names=["Fund"],
            factor_data=data,
            fit_method="LS",
        )

        # Should accept UMD instead of MOM
        assert len(model.factor_names) == 4
        assert "UMD" in model.factor_names

    def test_ff4_missing_momentum(self):
        """Test error when momentum factor is missing."""
        data = pl.DataFrame({
            "date": [date(2020, 1, 1)],
            "MKT": [0.01],
            "SMB": [0.005],
            "HML": [-0.002],
            # Missing MOM/UMD
            "Fund": [0.015],
        })

        with pytest.raises(ValueError, match="must contain 'MOM' or 'UMD'"):
            fit_ff4_model(
                asset_names=["Fund"],
                factor_data=data,
            )

    def test_ff4_multiple_assets(self):
        """Test FF4 with multiple assets."""
        n = 80
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        mkt = np.random.randn(n) * 0.02
        smb = np.random.randn(n) * 0.01
        hml = np.random.randn(n) * 0.01
        mom = np.random.randn(n) * 0.015

        data = pl.DataFrame({
            "date": dates,
            "MKT": mkt,
            "SMB": smb,
            "HML": hml,
            "MOM": mom,
            "Fund1": 0.8 * mkt + 0.3 * smb + 0.1 * hml + 0.2 * mom + np.random.randn(n) * 0.01,
            "Fund2": 1.2 * mkt - 0.2 * smb + 0.5 * hml - 0.1 * mom + np.random.randn(n) * 0.015,
        })

        model = fit_ff4_model(
            asset_names=["Fund1", "Fund2"],
            factor_data=data,
            fit_method="LS",
        )

        assert model.beta.shape == (2, 4)
        assert len(model.asset_names) == 2

    def test_ff4_robust(self):
        """Test FF4 with robust regression."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        data = pl.DataFrame({
            "date": dates,
            "MKT": np.random.randn(n) * 0.02,
            "SMB": np.random.randn(n) * 0.01,
            "HML": np.random.randn(n) * 0.01,
            "MOM": np.random.randn(n) * 0.015,
            "Fund": np.random.randn(n) * 0.025,
        })

        model = fit_ff4_model(
            asset_names=["Fund"],
            factor_data=data,
            fit_method="Robust",
            family="bisquare",
        )

        assert model.fit_method == "Robust"
        assert model.beta.shape == (1, 4)


class TestWrapperConsistency:
    """Tests for consistency across wrapper functions."""

    def test_all_wrappers_use_fit_tsfm(self):
        """Verify all wrappers ultimately call fit_tsfm."""
        n = 60
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        # Common data
        mkt = np.random.randn(n) * 0.02
        data_base = pl.DataFrame({
            "date": dates,
            "Asset1": 0.7 * mkt + np.random.randn(n) * 0.01,
            "Market": mkt,
        })

        # MT model
        mt_model = fit_tsfm_mt(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data_base,
            fit_method="LS",
        )
        assert "wrapper" in mt_model.call_info
        assert mt_model.call_info["wrapper"] == "fit_tsfm_mt"

        # UpDn model
        updn_result = fit_tsfm_up_dn(
            asset_names=["Asset1"],
            mkt_name="Market",
            data=data_base,
            fit_method="LS",
        )
        assert "wrapper" in updn_result.call_info
        assert updn_result.call_info["wrapper"] == "fit_tsfm_up_dn"

        # FF3 model
        ff3_data = data_base.with_columns([
            pl.Series("MKT", mkt),
            pl.Series("SMB", np.random.randn(n) * 0.01),
            pl.Series("HML", np.random.randn(n) * 0.01),
        ])
        ff3_model = fit_ff3_model(
            asset_names=["Asset1"],
            factor_data=ff3_data,
            fit_method="LS",
        )
        assert "wrapper" in ff3_model.call_info
        assert ff3_model.call_info["wrapper"] == "fit_ff3_model"

        # FF4 model
        ff4_data = ff3_data.with_columns([
            pl.Series("MOM", np.random.randn(n) * 0.015),
        ])
        ff4_model = fit_ff4_model(
            asset_names=["Asset1"],
            factor_data=ff4_data,
            fit_method="LS",
        )
        assert "wrapper" in ff4_model.call_info
        assert ff4_model.call_info["wrapper"] == "fit_ff4_model"
