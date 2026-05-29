"""
Data models for time series factor models.

This module defines the core data structures used throughout the package,
including TsfmModel (fitted model results) and TsfmControl (control parameters).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl


@dataclass
class TsfmModel:
    """
    Time series factor model results.

    This class stores the results from fitting a time series factor model using
    fit_tsfm(). It contains fitted model objects, estimated parameters (alpha, beta),
    goodness-of-fit statistics, residuals, and the original data.

    Attributes:
        asset_fit: Dictionary mapping asset names to fitted regression objects.
                   Each object is statsmodels OLS/RLM/WLS or sklearn LassoLarsCV.
        alpha: Estimated intercepts (N x 1 array where N = number of assets).
        beta: Estimated factor loadings (N x K matrix where K = number of factors).
        r_squared: R-squared values for each asset (length-N array).
        resid_sd: Residual standard deviations for each asset (length-N array).
        residuals: Residual matrix (N assets x T time periods).
        data: Original time series data as Polars DataFrame.
        asset_names: List of syntactically valid asset names.
        factor_names: List of syntactically valid factor names.
        fit_method: Estimation method used ("LS", "DLS", or "Robust").
        variable_selection: Variable selection method ("none", "stepwise", "subsets", "lars").
        rf_name: Name of risk-free rate column (if used for excess returns).
        mkt_name: Name of market return column (if used for market timing).
        call_info: Dictionary storing the original function call parameters.
        fitted_values: Optional fitted values matrix (N x T), populated for LARS.
    """

    asset_fit: Dict[str, Any]
    alpha: np.ndarray
    beta: np.ndarray
    r_squared: np.ndarray
    resid_sd: np.ndarray
    residuals: np.ndarray
    data: pl.DataFrame
    asset_names: List[str]
    factor_names: List[str]
    fit_method: Literal["LS", "DLS", "Robust"]
    variable_selection: Literal["none", "stepwise", "subsets", "lars"]
    rf_name: Optional[str] = None
    mkt_name: Optional[str] = None
    call_info: Dict[str, Any] = field(default_factory=dict)
    fitted_values: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate dimensions after initialization."""
        n_assets = len(self.asset_names)
        n_factors = len(self.factor_names)

        if self.alpha.shape[0] != n_assets:
            raise ValueError(f"alpha shape mismatch: expected {n_assets}, got {self.alpha.shape[0]}")
        if self.beta.shape != (n_assets, n_factors):
            raise ValueError(
                f"beta shape mismatch: expected ({n_assets}, {n_factors}), "
                f"got {self.beta.shape}"
            )
        if len(self.r_squared) != n_assets:
            raise ValueError(
                f"r_squared length mismatch: expected {n_assets}, got {len(self.r_squared)}"
            )
        if len(self.resid_sd) != n_assets:
            raise ValueError(
                f"resid_sd length mismatch: expected {n_assets}, got {len(self.resid_sd)}"
            )


@dataclass
class TsfmUpDn:
    """
    Up/Down market time series factor model results.

    This class stores results from fitting separate models for up and down markets
    using fit_tsfm_up_dn(). It contains two TsfmModel objects (one for each regime)
    and the market classification flag.

    Attributes:
        up_model: TsfmModel fitted on up market periods (excess market return >= 0).
        down_model: TsfmModel fitted on down market periods (excess market return < 0).
        up_market_flag: Boolean array indicating up market periods (length T).
        market_name: Name of the market return column used for regime classification.
        call_info: Dictionary storing the original function call parameters.
    """

    up_model: "TsfmModel"
    down_model: "TsfmModel"
    up_market_flag: np.ndarray
    market_name: str
    call_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TsfmControl:
    """
    Control parameters for fit_tsfm().

    This class consolidates all control parameters for the various estimation methods
    and variable selection procedures used in fit_tsfm(). It provides validated defaults
    and ensures compatibility between options.

    Parameters are organized by the internal function they control:
    - DLS-specific: decay, weights
    - lm (LS): model, x, y, qr
    - lmrobdetMM (Robust): nrep, bb, efficiency, tuning.chi, tuning.psi, family, etc.
    - step (Stepwise): scope, scale, direction, trace, steps, k
    - regsubsets (Subsets): nvmax, force.in, force.out, method, really.big
    - lars: type, normalize, eps, max.steps, use.Gram
    - cv.lars: K, cv.trace, mode

    Attributes:
        # DLS weights
        decay: Decay factor for discounted least squares (0 < decay <= 1). Default 0.95.
        weights: Optional weight vector for LS/Robust/Subsets regression.

        # lm control (LS method)
        model: If True, return model frame in lm object.
        x: If True, return model matrix in lm object.
        y: If True, return response vector in lm object.
        qr: If True, return QR decomposition in lm object.

        # lmrobdetMM control (Robust method)
        nrep: Number of random subsamples for robust estimation.
        tuning_chi: Tuning constant for M-scale computation.
        bb: Breakdown point for MM-estimator (0 < bb < 0.5). Default 0.5.
        tuning_psi: Tuning parameter for regression M-estimator.
        efficiency: Desired asymptotic efficiency (0 < efficiency < 1). Default 0.85.
        max_it: Maximum IRWLS iterations for MM-estimator.
        refine_tol: Convergence tolerance for S-estimator.
        rel_tol: Convergence tolerance for IRWLS iterations.
        refine_py: Number of Peña-Yohai refinement steps.
        solve_tol: Relative tolerance for matrix inversion.
        trace_lev: Verbosity level for MM-algorithm (0 = silent).
        compute_rd: Whether to compute robust leverage distances.
        family: Loss function family ("bisquare", "optimal", "modopt").
        corr_b: Whether to apply finite-sample correction to bb.
        split_type: How to split categorical/continuous variables.
        initial: Initial estimator for MM ("S" or "MS").

        # step control (Stepwise selection)
        scope: Formula or dict defining model search range.
        scale: Scale parameter for stepwise selection.
        direction: Stepwise direction ("both", "backward", "forward").
        trace: Verbosity level for stepwise/lars/cv.lars.
        steps: Maximum number of stepwise iterations.
        k: AIC penalty multiplier (k=2 for AIC, k=log(n) for BIC). Default 2.

        # regsubsets control (Subsets selection)
        nvmax: Maximum subset size to examine. Default 8.
        force_in: Variables to force into all models.
        force_out: Variables to force out of all models.
        method: Subset selection method ("exhaustive", "forward", "backward", "seqrep").
        really_big: Allow problems with >50 variables.

        # lars control
        lars_type: LARS variant ("lasso", "lar", "forward.stagewise", "stepwise").
        lars_normalize: Whether to standardize predictors. Default True.
        lars_eps: Numerical tolerance for lars.
        lars_max_steps: Maximum lars iterations.
        lars_use_gram: Whether to precompute Gram matrix.

        # cv.lars control
        lars_criterion: Model selection criterion ("Cp" or "cv"). Default "Cp".
        lars_cv_k: Number of CV folds for lars_criterion="cv". Default 10.
        lars_cv_trace: Print progress during CV.
        lars_cv_mode: CV mode ("fraction", "step", "norm"). Default "fraction".
    """

    # DLS weights
    decay: float = 0.95
    weights: Optional[np.ndarray] = None

    # lm control (LS)
    model: bool = True
    x: bool = False
    y: bool = False
    qr: bool = False

    # lmrobdetMM control (Robust)
    nrep: Optional[int] = None
    tuning_chi: Optional[float] = None
    bb: float = 0.5
    tuning_psi: Optional[float] = None
    efficiency: float = 0.85
    max_it: int = 100
    refine_tol: float = 1e-7
    rel_tol: float = 1e-7
    refine_py: int = 10
    solve_tol: float = 1e-7
    trace_lev: int = 0
    compute_rd: bool = False
    family: Literal["bisquare", "huber", "hampel", "andrews"] = "bisquare"
    corr_b: bool = True
    split_type: str = "f"
    initial: Literal["S", "MS"] = "S"

    # step control (Stepwise)
    scope: Optional[Union[str, Dict[str, str]]] = None
    scale: float = 0.0
    direction: Literal["both", "backward", "forward"] = "both"
    trace: int = 0
    steps: int = 1000
    k: float = 2.0

    # Convenience aliases for stepwise
    step_direction: Optional[Literal["both", "backward", "forward"]] = None
    step_criterion: Literal["aic", "bic"] = "bic"

    # regsubsets control (Subsets)
    nvmin: int = 1
    nvmax: int = 8
    force_in: Optional[List[int]] = None
    force_out: Optional[List[int]] = None
    method: Literal["exhaustive", "forward", "backward", "seqrep"] = "exhaustive"
    really_big: bool = False

    # lars control
    lars_type: Literal["lasso", "lar", "forward.stagewise", "stepwise"] = "lasso"
    lars_normalize: bool = True
    lars_eps: float = np.finfo(float).eps
    lars_max_steps: Optional[int] = None
    lars_use_gram: bool = True

    # cv.lars control
    lars_criterion: Literal["Cp", "cv", "aic", "bic"] = "Cp"
    lars_cv_k: int = 10
    lars_cv_folds: Optional[int] = None  # Alias for lars_cv_k
    lars_cv_trace: bool = False
    lars_cv_mode: Literal["fraction", "step", "norm"] = "fraction"

    def __post_init__(self):
        """Validate control parameters after initialization."""
        # Handle convenience aliases
        if self.step_direction is not None:
            self.direction = self.step_direction

        # Set k based on step_criterion if not explicitly provided
        # (this is a bit tricky since we can't detect if k was explicitly set)
        # For now, document that step_criterion overrides k
        if self.step_criterion == "bic":
            # BIC uses k = log(n), but we don't know n here
            # Document that user should set k=log(n) for BIC
            pass  # Keep default k=2 unless user changes it

        if self.lars_cv_folds is not None:
            self.lars_cv_k = self.lars_cv_folds

        # Validate decay
        if not (0 < self.decay <= 1):
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")

        # Validate bb (breakdown point)
        if not (0 < self.bb <= 0.5):
            raise ValueError(f"bb must be in (0, 0.5], got {self.bb}")

        # Validate efficiency
        if not (0 < self.efficiency < 1):
            raise ValueError(f"efficiency must be in (0, 1), got {self.efficiency}")

        # Validate k (AIC penalty)
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")

        # Validate nvmin and nvmax
        if self.nvmin < 1:
            raise ValueError(f"nvmin must be >= 1, got {self.nvmin}")
        if self.nvmax < 1:
            raise ValueError(f"nvmax must be >= 1, got {self.nvmax}")
        if self.nvmin > self.nvmax:
            raise ValueError(
                f"nvmin ({self.nvmin}) must be <= nvmax ({self.nvmax})"
            )

        # Validate lars_cv_k
        if self.lars_cv_k < 2:
            raise ValueError(f"lars_cv_k must be >= 2, got {self.lars_cv_k}")

        # Validate weights if provided
        if self.weights is not None:
            if not isinstance(self.weights, np.ndarray):
                raise TypeError("weights must be a numpy array")
            if np.any(self.weights < 0):
                raise ValueError("weights must be non-negative")
            if np.sum(self.weights == 0) == len(self.weights):
                raise ValueError("weights cannot all be zero")
