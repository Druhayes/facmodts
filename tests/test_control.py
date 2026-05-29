"""
Tests for control parameter handling.

This module tests control parameter validation and extraction functions.
"""

import numpy as np
import pytest

from facmodts.control import (
    fit_tsfm_control,
    get_robust_control_params,
    get_stepwise_control_params,
    get_subsets_control_params,
    get_lars_control_params,
)
from facmodts.models import TsfmControl


class TestFitTsfmControl:
    """Tests for fit_tsfm_control function."""

    def test_default_control(self):
        """Test creation with default values."""
        ctrl = fit_tsfm_control()

        assert ctrl.decay == 0.95
        assert ctrl.weights is None
        assert ctrl.bb == 0.5
        assert ctrl.efficiency == 0.85
        assert ctrl.family == "bisquare"
        assert ctrl.direction == "both"
        assert ctrl.nvmax == 8

    def test_dls_control(self):
        """Test DLS-specific parameters."""
        ctrl = fit_tsfm_control(decay=0.90)

        assert ctrl.decay == 0.90

    def test_robust_control(self):
        """Test robust regression parameters."""
        ctrl = fit_tsfm_control(
            bb=0.4,
            efficiency=0.90,
            family="huber",
            max_it=50
        )

        assert ctrl.bb == 0.4
        assert ctrl.efficiency == 0.90
        assert ctrl.family == "huber"
        assert ctrl.max_it == 50

    def test_stepwise_control(self):
        """Test stepwise selection parameters."""
        ctrl = fit_tsfm_control(
            direction="backward",
            k=np.log(100),
            steps=5
        )

        assert ctrl.direction == "backward"
        assert ctrl.k == np.log(100)
        assert ctrl.steps == 5

    def test_subsets_control(self):
        """Test subset selection parameters."""
        ctrl = fit_tsfm_control(
            nvmax=5,
            method="forward",
            force_in=[0, 1],
            really_big=True
        )

        assert ctrl.nvmax == 5
        assert ctrl.method == "forward"
        assert ctrl.force_in == [0, 1]
        assert ctrl.really_big is True

    def test_lars_control(self):
        """Test LARS parameters."""
        ctrl = fit_tsfm_control(
            lars_type="lasso",
            lars_criterion="cv",
            lars_cv_k=5,
            lars_normalize=True,
            lars_max_steps=100
        )

        assert ctrl.lars_type == "lasso"
        assert ctrl.lars_criterion == "cv"
        assert ctrl.lars_cv_k == 5
        assert ctrl.lars_normalize is True
        assert ctrl.lars_max_steps == 100

    def test_invalid_parameter_error(self):
        """Test error for invalid parameter."""
        with pytest.raises(TypeError, match="Invalid control parameter"):
            fit_tsfm_control(invalid_param="value")

    def test_empty_weights_error(self):
        """Test error for empty weights array."""
        # Empty array triggers "cannot all be zero" from TsfmControl validation
        with pytest.raises(ValueError, match="weights cannot all be zero"):
            fit_tsfm_control(weights=np.array([]))

    def test_weights_too_short_error(self):
        """Test error when weights array has insufficient length."""
        # Single zero weight
        with pytest.raises(ValueError, match="weights cannot all be zero"):
            fit_tsfm_control(weights=np.array([0.0]))

    def test_force_in_out_overlap_error(self):
        """Test error when force_in and force_out overlap."""
        with pytest.raises(ValueError, match="cannot contain the same variables"):
            fit_tsfm_control(
                force_in=[0, 1, 2],
                force_out=[1, 3]  # 1 appears in both
            )

    def test_nvmax_less_than_force_in_error(self):
        """Test error when nvmax < len(force_in)."""
        with pytest.raises(ValueError, match="nvmax.*must be >= length of force_in"):
            fit_tsfm_control(
                nvmax=2,
                force_in=[0, 1, 2]  # 3 forced in but nvmax=2
            )

    def test_scope_missing_keys_error(self):
        """Test error for scope dict missing required keys."""
        with pytest.raises(ValueError, match="scope dict must contain"):
            fit_tsfm_control(
                scope={"upper": "~ .", "wrong_key": "~ 1"}
            )

    def test_lars_max_steps_invalid_error(self):
        """Test error for invalid lars_max_steps."""
        with pytest.raises(ValueError, match="lars_max_steps must be >= 1"):
            fit_tsfm_control(lars_max_steps=0)

        with pytest.raises(ValueError, match="lars_max_steps must be >= 1"):
            fit_tsfm_control(lars_max_steps=-5)

    def test_decay_and_weights_warning(self):
        """Test warning when both decay and weights specified."""
        with pytest.warns(UserWarning, match="Both decay and weights specified"):
            fit_tsfm_control(
                decay=0.90,
                weights=np.array([1.0, 1.0, 1.0])
            )

    def test_valid_scope_dict(self):
        """Test valid scope dictionary."""
        ctrl = fit_tsfm_control(
            scope={"upper": "~ .", "lower": "~ 1"}
        )

        assert ctrl.scope["upper"] == "~ ."
        assert ctrl.scope["lower"] == "~ 1"

    def test_force_in_without_force_out(self):
        """Test force_in without force_out (should work)."""
        ctrl = fit_tsfm_control(
            force_in=[0, 1],
            nvmax=5
        )

        assert ctrl.force_in == [0, 1]
        assert ctrl.force_out is None

    def test_force_out_without_force_in(self):
        """Test force_out without force_in (should work)."""
        ctrl = fit_tsfm_control(
            force_out=[2, 3]
        )

        assert ctrl.force_out == [2, 3]
        assert ctrl.force_in is None


class TestGetRobustControlParams:
    """Tests for get_robust_control_params."""

    def test_extract_robust_params(self):
        """Test extraction of robust parameters."""
        ctrl = fit_tsfm_control(
            bb=0.4,
            efficiency=0.90,
            family="huber",
            max_it=50
        )

        params = get_robust_control_params(ctrl)

        assert params["bb"] == 0.4
        assert params["efficiency"] == 0.90
        assert params["family"] == "huber"
        assert params["max_it"] == 50

    def test_all_robust_params_present(self):
        """Test that all robust params are in returned dict."""
        ctrl = fit_tsfm_control()
        params = get_robust_control_params(ctrl)

        expected_keys = [
            "nrep", "tuning_chi", "bb", "tuning_psi", "efficiency",
            "max_it", "refine_tol", "rel_tol", "refine_py", "solve_tol",
            "trace_lev", "compute_rd", "family", "corr_b", "split_type",
            "initial"
        ]

        for key in expected_keys:
            assert key in params


class TestGetStepwiseControlParams:
    """Tests for get_stepwise_control_params."""

    def test_extract_stepwise_params(self):
        """Test extraction of stepwise parameters."""
        ctrl = fit_tsfm_control(
            direction="backward",
            k=np.log(100),
            steps=5
        )

        params = get_stepwise_control_params(ctrl)

        assert params["direction"] == "backward"
        assert params["k"] == np.log(100)
        assert params["steps"] == 5

    def test_all_stepwise_params_present(self):
        """Test that all stepwise params are in returned dict."""
        ctrl = fit_tsfm_control()
        params = get_stepwise_control_params(ctrl)

        expected_keys = ["scope", "scale", "direction", "trace", "steps", "k"]

        for key in expected_keys:
            assert key in params


class TestGetSubsetsControlParams:
    """Tests for get_subsets_control_params."""

    def test_extract_subsets_params(self):
        """Test extraction of subsets parameters."""
        ctrl = fit_tsfm_control(
            nvmax=5,
            method="forward",
            force_in=[0, 1]
        )

        params = get_subsets_control_params(ctrl)

        assert params["nvmax"] == 5
        assert params["method"] == "forward"
        assert params["force_in"] == [0, 1]

    def test_all_subsets_params_present(self):
        """Test that all subsets params are in returned dict."""
        ctrl = fit_tsfm_control()
        params = get_subsets_control_params(ctrl)

        expected_keys = [
            "nvmax", "force_in", "force_out", "method",
            "really_big", "weights"
        ]

        for key in expected_keys:
            assert key in params


class TestGetLarsControlParams:
    """Tests for get_lars_control_params."""

    def test_extract_lars_params(self):
        """Test extraction of LARS parameters."""
        ctrl = fit_tsfm_control(
            lars_type="lasso",
            lars_criterion="cv",
            lars_cv_k=5
        )

        params = get_lars_control_params(ctrl)

        assert params["lars_type"] == "lasso"
        assert params["lars_criterion"] == "cv"
        assert params["lars_cv_k"] == 5

    def test_all_lars_params_present(self):
        """Test that all LARS params are in returned dict."""
        ctrl = fit_tsfm_control()
        params = get_lars_control_params(ctrl)

        expected_keys = [
            "lars_type", "lars_normalize", "lars_eps", "lars_max_steps",
            "lars_use_gram", "lars_criterion", "lars_cv_k",
            "lars_cv_trace", "lars_cv_mode", "trace"
        ]

        for key in expected_keys:
            assert key in params
