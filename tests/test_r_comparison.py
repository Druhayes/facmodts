"""
R comparison tests for facmodts package.

These tests compare Python implementation outputs with R facmodTS package
to ensure numerical equivalence. Requires rpy2 and R with facmodTS installed.

To run these tests:
    pytest tests/test_r_comparison.py -v

To skip R comparison tests:
    pytest tests/ -m "not r_comparison"
"""

import numpy as np
import pytest

from facmodts import fit_tsfm

# All tests in this module require R
pytestmark = pytest.mark.r_comparison


class TestLSComparison:
    """Compare LS regression results with R facmodTS."""

    def test_ls_beta_comparison_single_asset(self, stocks_factors_subset, r_facmodts, tolerances):
        """Compare beta coefficients for single asset LS regression."""
        # Get first asset
        assets = stocks_factors_subset["TickerLast"].unique().to_list()
        asset = [assets[0]]
        factors = ["BP", "Beta60M", "PM12M1M"]

        # Python model
        py_model = fit_tsfm(
            data=stocks_factors_subset,
            asset_names=asset,
            factor_names=factors,
            fit_method="LS"
        )

        # R model (CRITICAL: use dot notation for R parameters!)
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        r_data = stocks_factors_subset.to_pandas()
        with ro.conversion.localconverter(ro.default_converter):
            r_df = ro.conversion.py2rpy(r_data)

        # Call R fitTsfm with proper parameter names (dots, not underscores!)
        r_model = r_facmodts.fitTsfm(**{
            "asset.names": ro.StrVector(asset),
            "factor.names": ro.StrVector(factors),
            "data": r_df,
            "fit.method": "LS",
            "variable.selection": "none"
        })

        # Extract R beta (inspect structure first)
        r_beta = np.array(r_model.rx2("beta"))

        # Compare beta coefficients
        np.testing.assert_allclose(
            py_model.beta,
            r_beta,
            **tolerances["beta"],
            err_msg=f"Beta mismatch for {asset[0]}"
        )

    def test_ls_multiple_assets_comparison(self, stocks_factors_subset, r_facmodts, tolerances):
        """Compare beta, alpha, R² for multiple assets."""
        assets = stocks_factors_subset["TickerLast"].unique().to_list()[:10]  # First 10 assets
        factors = ["BP", "Beta60M"]

        # Python model
        py_model = fit_tsfm(
            data=stocks_factors_subset,
            asset_names=assets,
            factor_names=factors,
            fit_method="LS"
        )

        # R model
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        r_data = stocks_factors_subset.to_pandas()
        with ro.conversion.localconverter(ro.default_converter):
            r_df = ro.conversion.py2rpy(r_data)

        r_model = r_facmodts.fitTsfm(**{
            "asset.names": ro.StrVector(assets),
            "factor.names": ro.StrVector(factors),
            "data": r_df,
            "fit.method": "LS",
            "variable.selection": "none"
        })

        # Extract R results
        r_alpha = np.array(r_model.rx2("alpha")).flatten()
        r_beta = np.array(r_model.rx2("beta"))
        r_r2 = np.array(r_model.rx2("r2"))
        r_resid_sd = np.array(r_model.rx2("resid.sd"))

        # Compare alpha
        np.testing.assert_allclose(
            py_model.alpha,
            r_alpha,
            **tolerances["alpha"],
            err_msg="Alpha mismatch"
        )

        # Compare beta
        np.testing.assert_allclose(
            py_model.beta,
            r_beta,
            **tolerances["beta"],
            err_msg="Beta mismatch"
        )

        # Compare R²
        np.testing.assert_allclose(
            py_model.r_squared,
            r_r2,
            **tolerances["r_squared"],
            err_msg="R² mismatch"
        )

        # Compare residual SD
        np.testing.assert_allclose(
            py_model.resid_sd,
            r_resid_sd,
            **tolerances["resid_sd"],
            err_msg="Residual SD mismatch"
        )


class TestDLSComparison:
    """Compare DLS regression results with R facmodTS."""

    def test_dls_default_decay(self, stocks_factors_subset, r_facmodts, tolerances):
        """Compare DLS with default decay factor (0.95)."""
        assets = stocks_factors_subset["TickerLast"].unique().to_list()[:5]
        factors = ["BP", "Beta60M"]

        # Python model
        py_model = fit_tsfm(
            data=stocks_factors_subset,
            asset_names=assets,
            factor_names=factors,
            fit_method="DLS"
        )

        # R model
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        r_data = stocks_factors_subset.to_pandas()
        with ro.conversion.localconverter(ro.default_converter):
            r_df = ro.conversion.py2rpy(r_data)

        r_model = r_facmodts.fitTsfm(**{
            "asset.names": ro.StrVector(assets),
            "factor.names": ro.StrVector(factors),
            "data": r_df,
            "fit.method": "DLS",
            "variable.selection": "none"
        })

        # Extract and compare
        r_beta = np.array(r_model.rx2("beta"))
        np.testing.assert_allclose(
            py_model.beta,
            r_beta,
            **tolerances["beta"],
            err_msg="DLS beta mismatch"
        )

    def test_dls_custom_decay(self, stocks_factors_subset, r_facmodts, tolerances):
        """Compare DLS with custom decay factor."""
        assets = stocks_factors_subset["TickerLast"].unique().to_list()[:3]
        factors = ["BP"]
        decay = 0.90

        # Python model
        py_model = fit_tsfm(
            data=stocks_factors_subset,
            asset_names=assets,
            factor_names=factors,
            fit_method="DLS",
            decay=decay
        )

        # R model
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        r_data = stocks_factors_subset.to_pandas()
        with ro.conversion.localconverter(ro.default_converter):
            r_df = ro.conversion.py2rpy(r_data)

        r_model = r_facmodts.fitTsfm(**{
            "asset.names": ro.StrVector(assets),
            "factor.names": ro.StrVector(factors),
            "data": r_df,
            "fit.method": "DLS",
            "variable.selection": "none",
            "decay": decay
        })

        # Extract and compare
        r_beta = np.array(r_model.rx2("beta"))
        np.testing.assert_allclose(
            py_model.beta,
            r_beta,
            **tolerances["beta"],
            err_msg=f"DLS beta mismatch with decay={decay}"
        )


class TestResidualComparison:
    """Compare residuals and fitted values."""

    def test_residuals_match(self, stocks_factors_subset, r_facmodts, tolerances):
        """Verify residuals match between Python and R."""
        assets = stocks_factors_subset["TickerLast"].unique().to_list()[:3]
        factors = ["BP", "Beta60M"]

        # Python model
        py_model = fit_tsfm(
            data=stocks_factors_subset,
            asset_names=assets,
            factor_names=factors,
            fit_method="LS"
        )

        # R model
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        r_data = stocks_factors_subset.to_pandas()
        with ro.conversion.localconverter(ro.default_converter):
            r_df = ro.conversion.py2rpy(r_data)

        r_model = r_facmodts.fitTsfm(**{
            "asset.names": ro.StrVector(assets),
            "factor.names": ro.StrVector(factors),
            "data": r_df,
            "fit.method": "LS",
            "variable.selection": "none"
        })

        # Compare residuals for first asset
        asset_name = assets[0]
        py_resid = py_model.asset_fit[asset_name].resid
        r_resid = np.array(r_model.rx2("asset.fit").rx2(asset_name).rx2("residuals"))

        np.testing.assert_allclose(
            py_resid,
            r_resid.flatten(),
            **tolerances["residuals"],
            err_msg=f"Residuals mismatch for {asset_name}"
        )


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_factor(self, stocks_factors_subset, r_facmodts, tolerances):
        """Test with a single factor."""
        assets = stocks_factors_subset["TickerLast"].unique().to_list()[:5]
        factors = ["BP"]

        # Python model
        py_model = fit_tsfm(
            data=stocks_factors_subset,
            asset_names=assets,
            factor_names=factors,
            fit_method="LS"
        )

        # R model
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        r_data = stocks_factors_subset.to_pandas()
        with ro.conversion.localconverter(ro.default_converter):
            r_df = ro.conversion.py2rpy(r_data)

        r_model = r_facmodts.fitTsfm(**{
            "asset.names": ro.StrVector(assets),
            "factor.names": ro.StrVector(factors),
            "data": r_df,
            "fit.method": "LS",
            "variable.selection": "none"
        })

        # Compare
        r_beta = np.array(r_model.rx2("beta"))
        np.testing.assert_allclose(
            py_model.beta,
            r_beta,
            **tolerances["beta"],
            err_msg="Beta mismatch for single factor model"
        )

    def test_many_factors(self, stocks_factors_subset, r_facmodts, tolerances):
        """Test with many factors."""
        assets = stocks_factors_subset["TickerLast"].unique().to_list()[:3]
        factors = ["BP", "Beta60M", "PM12M1M", "PM1M1M"]  # 4 factors

        # Python model
        py_model = fit_tsfm(
            data=stocks_factors_subset,
            asset_names=assets,
            factor_names=factors,
            fit_method="LS"
        )

        # R model
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        r_data = stocks_factors_subset.to_pandas()
        with ro.conversion.localconverter(ro.default_converter):
            r_df = ro.conversion.py2rpy(r_data)

        r_model = r_facmodts.fitTsfm(**{
            "asset.names": ro.StrVector(assets),
            "factor.names": ro.StrVector(factors),
            "data": r_df,
            "fit.method": "LS",
            "variable.selection": "none"
        })

        # Compare
        r_beta = np.array(r_model.rx2("beta"))
        np.testing.assert_allclose(
            py_model.beta,
            r_beta,
            **tolerances["beta"],
            err_msg="Beta mismatch for multi-factor model"
        )
