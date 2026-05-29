"""
facmodts: Time Series Factor Models for Portfolio Construction and Risk Analysis

Python implementation of the facmodTS R package for fitting time series factor models
using robust methods. Companion package to "Robust Portfolio Construction and Risk Analysis"
(Springer, 2025).

Main Functions:
    fit_tsfm: Fit time series factor model using LS, DLS, or Robust regression
    fit_tsfm_control: Create control parameters for fit_tsfm
    fm_cov: Compute factor model covariance matrix
    fm_sd_decomp: Decompose standard deviation by factor contributions
    fm_var_decomp: Decompose Value-at-Risk by factor contributions
    fm_es_decomp: Decompose Expected Shortfall by factor contributions
    pa_fm: Performance attribution by factors

Classes:
    TsfmModel: Fitted time series factor model
    TsfmControl: Control parameters for fitting
    TsfmUpDn: Up/down market models

Examples:
    >>> import polars as pl
    >>> from facmodts import fit_tsfm
    >>>
    >>> # Load data
    >>> data = pl.read_csv("returns.csv")
    >>>
    >>> # Fit LS model
    >>> model = fit_tsfm(
    ...     data=data,
    ...     asset_names=["Asset1", "Asset2"],
    ...     factor_names=["Factor1", "Factor2"],
    ...     fit_method="LS"
    ... )
    >>>
    >>> # Extract coefficients
    >>> print(model.beta)  # Factor loadings
"""

__version__ = "0.1.1"
__author__ = "Python Conversion Team"

# Core fitting
from .fitting import fit_tsfm
from .control import fit_tsfm_control

# Data models
from .models import TsfmModel, TsfmControl, TsfmUpDn

# Robust regression
from .robust import fit_rlm, get_robust_norm, compute_robust_scale, robust_r_squared

# Variable selection
from .variable_selection import select_stepwise, select_all_subsets, select_lars

# Risk decomposition
from .decomposition import fm_cov, fm_sd_decomp, fm_var_decomp, fm_es_decomp

# Wrappers
from .wrappers import fit_tsfm_mt, fit_tsfm_up_dn, fit_ff3_model, fit_ff4_model

# Performance attribution
from .performance import pa_fm, PaFm

# Reporting
from .reporting import (
    summary_tsfm,
    print_summary_tsfm,
    print_tsfm,
    predict_tsfm,
    print_pafm,
    summary_pafm,
)

# Plotting
from .plotting import plot_tsfm, plot_pafm, plot_tsfm_updn

# Utilities
from .utils import (
    make_syntactically_valid,
    compute_excess_returns,
    median_absolute_deviation,
)

__all__ = [
    # Main functions
    "fit_tsfm",
    "fit_tsfm_control",
    # Data models
    "TsfmModel",
    "TsfmControl",
    "TsfmUpDn",
    "PaFm",
    # Robust regression
    "fit_rlm",
    "get_robust_norm",
    "compute_robust_scale",
    "robust_r_squared",
    # Variable selection
    "select_stepwise",
    "select_all_subsets",
    "select_lars",
    # Risk decomposition
    "fm_cov",
    "fm_sd_decomp",
    "fm_var_decomp",
    "fm_es_decomp",
    # Wrappers
    "fit_tsfm_mt",
    "fit_tsfm_up_dn",
    "fit_ff3_model",
    "fit_ff4_model",
    # Performance attribution
    "pa_fm",
    # Reporting
    "summary_tsfm",
    "print_summary_tsfm",
    "print_tsfm",
    "predict_tsfm",
    "print_pafm",
    "summary_pafm",
    # Plotting
    "plot_tsfm",
    "plot_pafm",
    "plot_tsfm_updn",
    # Utilities
    "make_syntactically_valid",
    "compute_excess_returns",
    "median_absolute_deviation",
]

# Phase implementation status
_PHASE_STATUS = {
    "Phase 1": {
        "status": "Complete",
        "features": ["LS regression", "DLS regression", "Basic fitting"],
        "completed": ["Data models", "Control parameters", "Utilities"],
    },
    "Phase 2": {
        "status": "Complete",
        "features": ["Robust regression", "Variable selection (stepwise, subsets, LARS)"],
        "completed": ["M-estimators", "Stepwise selection", "Best subsets", "LARS/Lasso"],
    },
    "Phase 3": {
        "status": "Complete",
        "features": ["Risk decomposition (SD, VaR, ES)", "Factor model covariance"],
        "completed": ["fm_cov", "fm_sd_decomp", "fm_var_decomp", "fm_es_decomp"],
    },
    "Phase 4": {
        "status": "Complete",
        "features": ["Wrapper functions (MT, UpDn, FF models)"],
        "completed": ["fit_tsfm_mt", "fit_tsfm_up_dn", "fit_ff3_model", "fit_ff4_model"],
    },
    "Phase 5": {
        "status": "Complete",
        "features": ["Performance attribution", "Summary methods"],
        "completed": ["pa_fm", "summary_tsfm", "print_tsfm", "predict_tsfm", "summary_pafm", "print_pafm"],
    },
    "Phase 6": {
        "status": "In Progress",
        "features": ["Visualization", "Comprehensive testing"],
        "completed": ["plot_tsfm", "plot_pafm", "plot_tsfm_updn"],
    },
}
