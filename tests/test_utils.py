"""
Tests for utility functions.

This module tests utility functions for data validation, preprocessing,
and numerical computations.
"""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from facmodts.utils import (
    validate_column_names,
    make_syntactically_valid,
    compute_excess_returns,
    remove_incomplete_cases,
    compute_dls_weights,
    huber_loss,
    huber_psi,
    bisquare_psi,
    median_absolute_deviation,
    make_padded_dataframe,
    convert_date_column,
    check_positive_definite,
)


class TestValidateColumnNames:
    """Tests for validate_column_names."""

    def test_valid_columns(self):
        """Test with all valid columns."""
        data = pl.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9]
        })

        # Should not raise
        validate_column_names(data, ["A", "B"])
        validate_column_names(data, ["A", "B", "C"])

    def test_missing_columns_error(self):
        """Test error when columns are missing."""
        data = pl.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })

        with pytest.raises(ValueError, match="Columns not found"):
            validate_column_names(data, ["A", "C"])  # C doesn't exist

    def test_context_in_error_message(self):
        """Test that context appears in error message."""
        data = pl.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(ValueError, match="for asset.names"):
            validate_column_names(data, ["B"], context="asset.names")


class TestMakeSyntacticallyValid:
    """Tests for make_syntactically_valid."""

    def test_spaces_to_periods(self):
        """Test conversion of spaces to periods."""
        names = ["Asset 1", "Factor A", "Market"]
        result = make_syntactically_valid(names)

        assert result == ["Asset.1", "Factor.A", "Market"]

    def test_hyphens_to_periods(self):
        """Test conversion of hyphens to periods."""
        names = ["Asset-1", "Factor-A", "Market-Index"]
        result = make_syntactically_valid(names)

        assert result == ["Asset.1", "Factor.A", "Market.Index"]

    def test_mixed_special_chars(self):
        """Test mixed spaces and hyphens."""
        names = ["Asset 1-A", "Factor-B 2"]
        result = make_syntactically_valid(names)

        assert result == ["Asset.1.A", "Factor.B.2"]

    def test_already_valid(self):
        """Test names that are already valid."""
        names = ["Asset1", "Factor1", "Market"]
        result = make_syntactically_valid(names)

        assert result == names


class TestComputeExcessReturns:
    """Tests for compute_excess_returns."""

    def test_basic_excess_returns(self):
        """Test computation of excess returns."""
        data = pl.DataFrame({
            "Asset1": [0.05, 0.03, -0.02],
            "Asset2": [0.04, 0.02, -0.01],
            "Factor1": [0.03, 0.02, -0.01],
            "RF": [0.01, 0.01, 0.01]
        })

        result = compute_excess_returns(
            data,
            asset_names=["Asset1", "Asset2"],
            factor_names=["Factor1"],
            rf_name="RF"
        )

        # Check excess returns
        assert result["Asset1"][0] == pytest.approx(0.04)  # 0.05 - 0.01
        assert result["Asset2"][0] == pytest.approx(0.03)  # 0.04 - 0.01
        assert result["Factor1"][0] == pytest.approx(0.02)  # 0.03 - 0.01

    def test_excess_returns_preserves_other_columns(self):
        """Test that other columns are preserved."""
        data = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 1, 2)],
            "Asset1": [0.05, 0.03],
            "Factor1": [0.03, 0.02],
            "RF": [0.01, 0.01]
        })

        result = compute_excess_returns(
            data,
            asset_names=["Asset1"],
            factor_names=["Factor1"],
            rf_name="RF"
        )

        assert "date" in result.columns
        assert result["date"][0] == date(2020, 1, 1)


class TestRemoveIncompleteCases:
    """Tests for remove_incomplete_cases."""

    def test_remove_rows_with_nulls(self):
        """Test removal of rows with null values."""
        data = pl.DataFrame({
            "Asset1": [1.0, 2.0, None, 4.0],
            "Factor1": [0.1, 0.2, 0.3, None],
            "Factor2": [0.5, 0.6, 0.7, 0.8]
        })

        result = remove_incomplete_cases(
            data,
            asset_name="Asset1",
            factor_names=["Factor1", "Factor2"]
        )

        # Should keep only rows 0 and 1 (rows 2 and 3 have nulls)
        assert len(result) == 2
        assert result["Asset1"][0] == 1.0
        assert result["Asset1"][1] == 2.0

    def test_no_nulls(self):
        """Test with no null values."""
        data = pl.DataFrame({
            "Asset1": [1.0, 2.0, 3.0],
            "Factor1": [0.1, 0.2, 0.3]
        })

        result = remove_incomplete_cases(
            data,
            asset_name="Asset1",
            factor_names=["Factor1"]
        )

        assert len(result) == 3


class TestComputeDlsWeights:
    """Tests for compute_dls_weights."""

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        weights = compute_dls_weights(100, 0.95)

        assert np.sum(weights) == pytest.approx(1.0)

    def test_weights_increasing(self):
        """Test that weights increase (more recent = higher weight)."""
        weights = compute_dls_weights(10, 0.95)

        # Weights should increase monotonically
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]

    def test_decay_zero_point_five(self):
        """Test with decay=0.5."""
        weights = compute_dls_weights(5, 0.5)

        assert np.sum(weights) == pytest.approx(1.0)
        assert weights[-1] > weights[0]  # Last weight larger

    def test_decay_one(self):
        """Test with decay=1.0 (equal weights)."""
        weights = compute_dls_weights(5, 1.0)

        # All weights should be equal
        assert np.all(weights == pytest.approx(0.2))


class TestHuberLoss:
    """Tests for huber_loss."""

    def test_quadratic_for_small_residuals(self):
        """Test quadratic loss for |r| <= c."""
        residuals = np.array([-0.5, 0.0, 0.5])
        c = 1.345

        loss = huber_loss(residuals, c)

        # Should be r^2 / 2 for |r| <= c
        expected = np.array([0.125, 0.0, 0.125])
        np.testing.assert_allclose(loss, expected)

    def test_linear_for_large_residuals(self):
        """Test linear loss for |r| > c."""
        residuals = np.array([-2.0, 2.0])
        c = 1.345

        loss = huber_loss(residuals, c)

        # Should be c|r| - c^2/2 for |r| > c
        expected = c * 2.0 - 0.5 * c * c
        np.testing.assert_allclose(loss, [expected, expected])


class TestHuberPsi:
    """Tests for huber_psi."""

    def test_identity_for_small_residuals(self):
        """Test psi(r) = r for |r| <= c."""
        residuals = np.array([-0.5, 0.0, 0.5, 1.0])
        c = 1.345

        psi = huber_psi(residuals, c)

        # Should equal residuals for |r| <= c
        np.testing.assert_allclose(psi, residuals)

    def test_constant_for_large_residuals(self):
        """Test psi(r) = c*sign(r) for |r| > c."""
        residuals = np.array([-2.0, -3.0, 2.0, 3.0])
        c = 1.345

        psi = huber_psi(residuals, c)

        # Should be c*sign(r) for |r| > c
        expected = np.array([-c, -c, c, c])
        np.testing.assert_allclose(psi, expected)


class TestBisquarePsi:
    """Tests for bisquare_psi."""

    def test_zero_for_large_residuals(self):
        """Test psi(r) = 0 for |r| > c."""
        residuals = np.array([-5.0, -10.0, 5.0, 10.0])
        c = 4.685

        psi = bisquare_psi(residuals, c)

        # Should be zero for |r| > c
        np.testing.assert_allclose(psi, np.zeros(4))

    def test_nonzero_for_small_residuals(self):
        """Test psi(r) != 0 for |r| <= c."""
        residuals = np.array([-1.0, 0.0, 1.0])
        c = 4.685

        psi = bisquare_psi(residuals, c)

        # psi(0) should be 0
        assert psi[1] == 0.0

        # psi(±1) should be nonzero and symmetric
        assert psi[0] < 0
        assert psi[2] > 0
        assert abs(psi[0]) == pytest.approx(abs(psi[2]))


class TestMedianAbsoluteDeviation:
    """Tests for median_absolute_deviation."""

    def test_mad_normal_data(self):
        """Test MAD with normal data."""
        np.random.seed(42)
        x = np.random.randn(1000)

        mad = median_absolute_deviation(x)

        # For normal data, MAD should be close to std
        assert mad == pytest.approx(np.std(x), rel=0.1)

    def test_mad_with_outliers(self):
        """Test MAD robustness to outliers."""
        x = np.array([1, 2, 3, 4, 5, 100, 200])  # 100, 200 are outliers

        mad = median_absolute_deviation(x)
        std = np.std(x)

        # MAD should be much smaller than std due to outliers
        assert mad < std

    def test_mad_constant_array(self):
        """Test MAD of constant array."""
        x = np.ones(10)

        mad = median_absolute_deviation(x)

        # MAD of constant should be 0
        assert mad == 0.0


class TestMakePaddedDataframe:
    """Tests for make_padded_dataframe."""

    def test_dict_with_unequal_factors(self):
        """Test with dictionaries of unequal length."""
        coef_dict = {
            "Asset1": {"alpha": 0.01, "Factor1": 0.5, "Factor2": 0.3},
            "Asset2": {"alpha": 0.02, "Factor1": 0.7}  # Factor2 missing
        }

        matrix, rows, cols = make_padded_dataframe(coef_dict)

        assert rows == ["Asset1", "Asset2"]
        assert set(cols) == {"alpha", "Factor1", "Factor2"}

        # Asset2 should have NaN for Factor2
        factor2_idx = cols.index("Factor2")
        assert np.isnan(matrix[1, factor2_idx])

    def test_dict_with_equal_factors(self):
        """Test with dictionaries of equal length."""
        coef_dict = {
            "Asset1": {"alpha": 0.01, "Factor1": 0.5},
            "Asset2": {"alpha": 0.02, "Factor1": 0.7}
        }

        matrix, rows, cols = make_padded_dataframe(coef_dict)

        assert rows == ["Asset1", "Asset2"]
        assert set(cols) == {"alpha", "Factor1"}

        # No NaNs
        assert not np.any(np.isnan(matrix))

    def test_array_input(self):
        """Test with numpy array input."""
        coef_dict = {
            "Asset1": np.array([0.01, 0.5, 0.3]),
            "Asset2": np.array([0.02, 0.7, 0.4])
        }

        matrix, rows, cols = make_padded_dataframe(coef_dict)

        assert rows == ["Asset1", "Asset2"]
        assert len(cols) == 3


class TestConvertDateColumn:
    """Tests for convert_date_column."""

    def test_convert_string_to_date(self):
        """Test conversion of string to Date."""
        data = pl.DataFrame({
            "date": ["2020-01-01", "2020-01-02"],
            "value": [1, 2]
        })

        # Convert string to date first
        data = data.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

        result = convert_date_column(data)

        assert result["date"].dtype == pl.Date

    def test_already_date_type(self):
        """Test when column is already Date type."""
        dates = [date(2020, 1, 1), date(2020, 1, 2)]
        data = pl.DataFrame({
            "date": dates,
            "value": [1, 2]
        })

        result = convert_date_column(data)

        # Should remain Date type
        assert result["date"].dtype == pl.Date

    def test_no_date_column(self):
        """Test when date column doesn't exist."""
        data = pl.DataFrame({"value": [1, 2, 3]})

        result = convert_date_column(data)

        # Should return unchanged
        assert result.equals(data)


class TestCheckPositiveDefinite:
    """Tests for check_positive_definite."""

    def test_positive_definite_matrix(self):
        """Test with positive definite matrix."""
        # Identity matrix is positive definite
        matrix = np.eye(3)

        assert check_positive_definite(matrix) is True

    def test_non_positive_definite_matrix(self):
        """Test with non-positive definite matrix."""
        # Singular matrix (not positive definite)
        matrix = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9]
        ])

        assert check_positive_definite(matrix) is False

    def test_non_square_matrix_error(self):
        """Test error with non-square matrix."""
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        with pytest.raises(ValueError, match="must be square"):
            check_positive_definite(matrix)

    def test_symmetric_pd_matrix(self):
        """Test with symmetric positive definite matrix."""
        # Covariance matrix (symmetric PD)
        matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        assert check_positive_definite(matrix) is True

    def test_negative_definite_matrix(self):
        """Test with negative definite matrix."""
        matrix = -np.eye(3)

        assert check_positive_definite(matrix) is False
