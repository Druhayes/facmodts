"""
Control parameter handling for fit_tsfm().

This module provides the fit_tsfm_control() function which processes and validates
control parameters for time series factor model fitting.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from .models import TsfmControl


def fit_tsfm_control(**kwargs) -> TsfmControl:
    """
    Create and validate control parameters for fit_tsfm().

    This function processes optional control arguments and returns a TsfmControl
    object with validated parameters. Parameters not specified will use their
    default values as defined in the TsfmControl dataclass.

    All control parameters are validated for correctness and compatibility.
    Invalid values will raise ValueError or TypeError exceptions.

    Parameters:
        **kwargs: Control parameters. See TsfmControl docstring for full list.

    Returns:
        TsfmControl object with validated parameters.

    Raises:
        ValueError: If any parameter has an invalid value.
        TypeError: If any parameter has an incorrect type.

    Examples:
        >>> # Basic usage with defaults
        >>> ctrl = fit_tsfm_control()
        >>> ctrl.decay
        0.95

        >>> # DLS with custom decay
        >>> ctrl = fit_tsfm_control(decay=0.90)
        >>> ctrl.decay
        0.9

        >>> # Robust estimation with custom breakdown point
        >>> ctrl = fit_tsfm_control(bb=0.4, efficiency=0.90)
        >>> ctrl.bb
        0.4

        >>> # Stepwise selection with BIC
        >>> ctrl = fit_tsfm_control(k=np.log(100), direction="backward")
        >>> ctrl.direction
        'backward'

        >>> # LARS with cross-validation
        >>> ctrl = fit_tsfm_control(
        ...     lars_criterion="cv",
        ...     lars_cv_k=5,
        ...     lars_type="lasso"
        ... )
        >>> ctrl.lars_criterion
        'cv'
    """
    # Create TsfmControl with provided kwargs
    # The dataclass __post_init__ will handle validation
    try:
        control = TsfmControl(**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid control parameter: {e}")

    # Additional validation beyond __post_init__

    # Validate weights if provided
    if control.weights is not None:
        if len(control.weights) == 0:
            raise ValueError("weights must have length > 0")

    # Validate force_in/force_out compatibility
    if control.force_in is not None and control.force_out is not None:
        force_in_set = set(control.force_in)
        force_out_set = set(control.force_out)
        overlap = force_in_set.intersection(force_out_set)
        if overlap:
            raise ValueError(
                f"force_in and force_out cannot contain the same variables: {overlap}"
            )

    # Validate nvmax makes sense
    if control.force_in is not None:
        if len(control.force_in) > control.nvmax:
            raise ValueError(
                f"nvmax ({control.nvmax}) must be >= length of force_in "
                f"({len(control.force_in)})"
            )

    # Validate scope structure if provided
    if control.scope is not None:
        if isinstance(control.scope, dict):
            required_keys = {"upper", "lower"}
            if not required_keys.issubset(control.scope.keys()):
                raise ValueError(
                    f"scope dict must contain 'upper' and 'lower' keys, "
                    f"got {list(control.scope.keys())}"
                )

    # Validate lars_max_steps if provided
    if control.lars_max_steps is not None:
        if control.lars_max_steps < 1:
            raise ValueError(f"lars_max_steps must be >= 1, got {control.lars_max_steps}")

    # Warn if both decay and weights are specified (weights will be ignored for DLS)
    if control.decay != 0.95 and control.weights is not None:
        import warnings

        warnings.warn(
            "Both decay and weights specified. For fit_method='DLS', weights will be "
            "computed from decay and the provided weights will be ignored.",
            UserWarning,
        )

    return control


def get_robust_control_params(control: TsfmControl) -> Dict[str, Any]:
    """
    Extract robust regression control parameters from TsfmControl.

    This helper function creates a dictionary of parameters suitable for
    passing to statsmodels RLM or similar robust regression functions.

    Parameters:
        control: TsfmControl object.

    Returns:
        Dictionary of robust regression parameters.

    Examples:
        >>> ctrl = fit_tsfm_control(bb=0.4, efficiency=0.90)
        >>> params = get_robust_control_params(ctrl)
        >>> params['bb']
        0.4
    """
    params = {
        "nrep": control.nrep,
        "tuning_chi": control.tuning_chi,
        "bb": control.bb,
        "tuning_psi": control.tuning_psi,
        "efficiency": control.efficiency,
        "max_it": control.max_it,
        "refine_tol": control.refine_tol,
        "rel_tol": control.rel_tol,
        "refine_py": control.refine_py,
        "solve_tol": control.solve_tol,
        "trace_lev": control.trace_lev,
        "compute_rd": control.compute_rd,
        "family": control.family,
        "corr_b": control.corr_b,
        "split_type": control.split_type,
        "initial": control.initial,
    }
    return params


def get_stepwise_control_params(control: TsfmControl) -> Dict[str, Any]:
    """
    Extract stepwise selection control parameters from TsfmControl.

    Parameters:
        control: TsfmControl object.

    Returns:
        Dictionary of stepwise selection parameters.

    Examples:
        >>> ctrl = fit_tsfm_control(direction="backward", k=np.log(100))
        >>> params = get_stepwise_control_params(ctrl)
        >>> params['direction']
        'backward'
    """
    params = {
        "scope": control.scope,
        "scale": control.scale,
        "direction": control.direction,
        "trace": control.trace,
        "steps": control.steps,
        "k": control.k,
    }
    return params


def get_subsets_control_params(control: TsfmControl) -> Dict[str, Any]:
    """
    Extract subset selection control parameters from TsfmControl.

    Parameters:
        control: TsfmControl object.

    Returns:
        Dictionary of subset selection parameters.

    Examples:
        >>> ctrl = fit_tsfm_control(nvmax=5, method="forward")
        >>> params = get_subsets_control_params(ctrl)
        >>> params['nvmax']
        5
    """
    params = {
        "nvmax": control.nvmax,
        "force_in": control.force_in,
        "force_out": control.force_out,
        "method": control.method,
        "really_big": control.really_big,
        "weights": control.weights,  # regsubsets can use weights
    }
    return params


def get_lars_control_params(control: TsfmControl) -> Dict[str, Any]:
    """
    Extract LARS control parameters from TsfmControl.

    Parameters:
        control: TsfmControl object.

    Returns:
        Dictionary of LARS parameters.

    Examples:
        >>> ctrl = fit_tsfm_control(lars_type="lasso", lars_normalize=True)
        >>> params = get_lars_control_params(ctrl)
        >>> params['lars_type']
        'lasso'
    """
    params = {
        "lars_type": control.lars_type,
        "lars_normalize": control.lars_normalize,
        "lars_eps": control.lars_eps,
        "lars_max_steps": control.lars_max_steps,
        "lars_use_gram": control.lars_use_gram,
        "lars_criterion": control.lars_criterion,
        "lars_cv_k": control.lars_cv_k,
        "lars_cv_trace": control.lars_cv_trace,
        "lars_cv_mode": control.lars_cv_mode,
        "trace": control.trace,
    }
    return params
