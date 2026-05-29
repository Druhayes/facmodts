"""
Risk decomposition functions for factor models.

This module implements Euler decomposition for standard deviation (SD),
Value-at-Risk (VaR), and Expected Shortfall (ES) based on fitted factor models.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats

from .models import TsfmModel


def fm_cov(
    model: TsfmModel,
    factor_cov: Optional[np.ndarray] = None,
    use: str = "pairwise"
) -> np.ndarray:
    """
    Compute covariance matrix for asset returns from fitted factor model.

    The N x N covariance matrix of asset returns is given by:
        Cov(R) = B * Cov(F) * B' + D

    where B is the N x K matrix of factor betas and D is a diagonal matrix
    with residual variances sig(i)^2 along the diagonal.

    Parameters:
        model: Fitted TsfmModel object.
        factor_cov: Optional K x K factor covariance matrix. If None, computed
                    from factor returns in model.data.
        use: Method for computing covariances with missing values ("pairwise"
             or "complete"). Default "pairwise".

    Returns:
        N x N covariance matrix of asset returns.

    Raises:
        Warning: If covariance matrix is not positive definite.

    Examples:
        >>> model = fit_tsfm(data, asset_names=["A1", "A2"], factor_names=["F1", "F2"])
        >>> cov_matrix = fm_cov(model)
        >>> cov_matrix.shape
        (2, 2)
    """
    # Get beta matrix: N x K
    beta = model.beta.copy()
    beta[np.isnan(beta)] = 0  # Replace NAs with 0 for matrix multiplication

    # Get residual variances
    sig2_e = model.resid_sd ** 2

    # Compute factor covariance matrix if not provided
    if factor_cov is None:
        # Extract factor returns from data
        factor_df = model.data.select(model.factor_names).to_pandas()

        if use == "pairwise":
            factor_cov = factor_df.cov().values
        elif use == "complete":
            factor_df_complete = factor_df.dropna()
            factor_cov = factor_df_complete.cov().values
        else:
            raise ValueError(f"use must be 'pairwise' or 'complete', got '{use}'")
    else:
        # Validate dimensions
        K = len(model.factor_names)
        if factor_cov.shape != (K, K):
            raise ValueError(
                f"factor_cov must be {K}x{K}, got {factor_cov.shape}"
            )

    # Compute residual covariance matrix D
    if len(sig2_e) > 1:
        D_e = np.diag(sig2_e)
    else:
        D_e = sig2_e  # Scalar case

    # Compute covariance: B * Cov(F) * B' + D
    cov_fm = beta @ factor_cov @ beta.T + D_e

    # Check positive definiteness
    try:
        np.linalg.cholesky(cov_fm)
    except np.linalg.LinAlgError:
        import warnings
        warnings.warn("Covariance matrix is not positive definite!")

    return cov_fm


def fm_sd_decomp(
    model: TsfmModel,
    factor_cov: Optional[np.ndarray] = None,
    use: str = "pairwise"
) -> Dict[str, np.ndarray]:
    """
    Decompose standard deviation into factor contributions using Euler's theorem.

    By Euler's theorem, the standard deviation of asset i's return is:
        SD_i = sum(cSd_k) = sum(beta_star_k * mSd_k)

    where summation is across K factors + 1 residual term.

    The factor model is: R(t) = beta'*F(t) + e(t) = beta_star'*F_star(t)
    where beta_star = (beta, sig_e) and F_star = (F, z).

    Formulas:
        SD = sqrt(beta_star' * Cov(F_star) * beta_star)
        mSD = Cov(F_star) * beta_star / SD
        cSD = mSD * beta_star
        pcSD = 100 * cSD / SD

    Parameters:
        model: Fitted TsfmModel object.
        factor_cov: Optional K x K factor covariance matrix. If None, computed
                    from factor returns.
        use: Method for computing covariances ("pairwise" or "complete").

    Returns:
        Dictionary containing:
        - sd_fm: N-vector of factor model standard deviations
        - m_sd: N x (K+1) matrix of marginal contributions
        - c_sd: N x (K+1) matrix of component contributions
        - pc_sd: N x (K+1) matrix of percentage contributions

    Examples:
        >>> model = fit_tsfm(data, asset_names=["A1"], factor_names=["F1", "F2"])
        >>> decomp = fm_sd_decomp(model)
        >>> decomp['pc_sd']  # Percentage contributions
        array([[45.2, 38.1, 16.7]])  # Factor1, Factor2, Residual
    """
    # Get beta_star: N x (K+1)
    beta = model.beta.copy()
    beta[np.isnan(beta)] = 0
    beta_star = np.column_stack([beta, model.resid_sd])

    # Get factor covariance matrix
    if factor_cov is None:
        factor_df = model.data.select(model.factor_names).to_pandas()
        if use == "pairwise":
            factor_cov = factor_df.cov().values
        else:
            factor_df_complete = factor_df.dropna()
            factor_cov = factor_df_complete.cov().values

    # Build Cov(F_star): (K+1) x (K+1)
    K = len(model.factor_names)
    factor_star_cov = np.eye(K + 1)
    factor_star_cov[:K, :K] = factor_cov

    # Compute factor model SD: N-vector
    # SD = sqrt(sum_k beta_star_k * (Cov(F_star) * beta_star)_k)
    sd_fm = np.sqrt(np.sum(beta_star @ factor_star_cov * beta_star, axis=1))

    # Compute marginal SD: N x (K+1)
    # mSD = Cov(F_star) * beta_star / SD
    m_sd = (factor_star_cov @ beta_star.T).T / sd_fm[:, np.newaxis]

    # Compute component SD: N x (K+1)
    # cSD = mSD * beta_star
    c_sd = m_sd * beta_star

    # Compute percentage SD: N x (K+1)
    # pcSD = 100 * cSD / SD
    pc_sd = 100 * c_sd / sd_fm[:, np.newaxis]

    return {
        "sd_fm": sd_fm,
        "m_sd": m_sd,
        "c_sd": c_sd,
        "pc_sd": pc_sd
    }


def fm_var_decomp(
    model: TsfmModel,
    p: float = 0.05,
    type: Literal["np", "normal"] = "np",
    factor_cov: Optional[np.ndarray] = None,
    use: str = "pairwise"
) -> Dict[str, np.ndarray]:
    """
    Decompose Value-at-Risk into factor contributions using Euler's theorem.

    By Euler's theorem, VaR of asset i's return is:
        VaR_i = sum(cVaR_k) = sum(beta_star_k * mVaR_k)

    The marginal VaR is computed as the expected factor return conditional
    on the asset return being equal to its VaR. For non-parametric estimation,
    a triangular kernel is used following Epperlein & Smillie (2006).

    Parameters:
        model: Fitted TsfmModel object.
        p: Tail probability (default 0.05 for 5% VaR).
        type: "np" for non-parametric or "normal" for Gaussian VaR.
        factor_cov: Optional K x K factor covariance matrix (for normal type).
        use: Method for computing covariances.

    Returns:
        Dictionary containing:
        - var_fm: N-vector of factor model VaRs
        - n_exceed: N-vector of number of VaR exceedances
        - idx_exceed: List of arrays with exceedance indices per asset
        - m_var: N x (K+1) matrix of marginal contributions
        - c_var: N x (K+1) matrix of component contributions
        - pc_var: N x (K+1) matrix of percentage contributions

    References:
        Epperlein & Smillie (2006). Portfolio risk analysis: Cracking VAR with kernels.

    Examples:
        >>> model = fit_tsfm(data, asset_names=["A1"], factor_names=["F1", "F2"])
        >>> var_decomp = fm_var_decomp(model, p=0.05)
        >>> var_decomp['var_fm']  # 5% VaR
        array([-0.0234])
        >>> var_decomp['pc_var']  # Percentage contributions
        array([[52.3, 35.1, 12.6]])
    """
    N = len(model.asset_names)
    K = len(model.factor_names)

    # Get beta_star: N x (K+1)
    beta = model.beta.copy()
    beta[np.isnan(beta)] = 0
    beta_star = np.column_stack([beta, model.resid_sd])

    # Extract factor returns and compute standardized residuals
    factor_df = model.data.select(model.factor_names).to_pandas()
    factors = factor_df.values

    # Standardized residuals: residuals / resid_sd
    resid_std = model.residuals / model.resid_sd[:, np.newaxis]

    # Initialize arrays
    var_fm = np.full(N, np.nan)
    n_exceed = np.zeros(N, dtype=int)
    idx_exceed = []
    m_var = np.full((N, K + 1), np.nan)
    c_var = np.full((N, K + 1), np.nan)
    pc_var = np.full((N, K + 1), np.nan)

    if type == "normal":
        # Get factor covariance
        if factor_cov is None:
            if use == "pairwise":
                factor_cov = factor_df.cov().values
            else:
                factor_df_complete = factor_df.dropna()
                factor_cov = factor_df_complete.cov().values

        # Build Cov(F_star)
        factor_star_cov = np.eye(K + 1)
        factor_star_cov[:K, :K] = factor_cov

        # Factor expected returns (mean, 0 for residual)
        MU = np.concatenate([factor_df.mean().values, [0]])

        # SIGMA * Beta for normal mVaR computation
        SIGB = beta_star @ factor_star_cov

    # Loop over assets
    for i in range(N):
        # Get asset returns (extract from data)
        asset_name = model.asset_names[i]
        R = model.data[asset_name].to_numpy()

        # Remove NaNs
        valid_idx = ~np.isnan(R)
        R_valid = R[valid_idx]

        if len(R_valid) == 0:
            continue

        if type == "np":
            # Non-parametric VaR: p-quantile
            var_fm[i] = np.quantile(R_valid, p)

            # Get F_star: combine factors and standardized residual
            # Need to align with valid indices
            factors_valid = factors[valid_idx, :]
            resid_std_valid = resid_std[i, valid_idx]
            factor_star = np.column_stack([factors_valid, resid_std_valid])

            # Silverman's rule of thumb for bandwidth (triangular kernel)
            eps = 2.575 * np.std(R_valid, ddof=1) * (len(R_valid) ** (-0.2))

            # Triangular kernel weights
            k_weight = 1 - np.abs(R_valid - var_fm[i]) / eps
            k_weight[k_weight < 0] = 0

            # Marginal VaR: kernel-weighted expected factor returns
            if k_weight.sum() > 0:
                m_var[i, :] = np.average(factor_star, axis=0, weights=k_weight)
            else:
                # Fallback if no weights
                m_var[i, :] = np.mean(factor_star, axis=0)

        elif type == "normal":
            # Normal VaR
            var_fm[i] = (
                beta_star[i, :] @ MU +
                np.sqrt(beta_star[i, :] @ factor_star_cov @ beta_star[i, :]) *
                stats.norm.ppf(p)
            )

            # Marginal VaR for normal
            m_var[i, :] = (
                MU + SIGB[i, :] * stats.norm.ppf(p) / np.std(R_valid, ddof=1)
            )

        # Index of VaR exceedances
        exceed_mask = R_valid <= var_fm[i]
        idx_exceed.append(np.where(exceed_mask)[0])
        n_exceed[i] = int(exceed_mask.sum())

        # Correction factor to ensure sum(cVaR) = VaR
        component_sum = np.sum(m_var[i, :] * beta_star[i, :])
        if component_sum != 0:
            cf = var_fm[i] / component_sum
        else:
            cf = 1.0

        # Apply correction and compute components
        m_var[i, :] = cf * m_var[i, :]
        c_var[i, :] = m_var[i, :] * beta_star[i, :]
        pc_var[i, :] = 100 * c_var[i, :] / var_fm[i]

    return {
        "var_fm": var_fm,
        "n_exceed": n_exceed,
        "idx_exceed": idx_exceed,
        "m_var": m_var,
        "c_var": c_var,
        "pc_var": pc_var
    }


def fm_es_decomp(
    model: TsfmModel,
    p: float = 0.05,
    type: Literal["np", "normal"] = "np",
    factor_cov: Optional[np.ndarray] = None,
    use: str = "pairwise"
) -> Dict[str, np.ndarray]:
    """
    Decompose Expected Shortfall into factor contributions using Euler's theorem.

    By Euler's theorem, ES of asset i's return is:
        ES_i = sum(cES_k) = sum(beta_star_k * mES_k)

    The marginal ES is the expected factor return conditional on the asset
    return being less than or equal to its VaR.

    Parameters:
        model: Fitted TsfmModel object.
        p: Tail probability (default 0.05 for 5% ES).
        type: "np" for non-parametric or "normal" for Gaussian ES.
        factor_cov: Optional K x K factor covariance matrix (for normal type).
        use: Method for computing covariances.

    Returns:
        Dictionary containing:
        - es_fm: N-vector of factor model ES values
        - m_es: N x (K+1) matrix of marginal contributions
        - c_es: N x (K+1) matrix of component contributions
        - pc_es: N x (K+1) matrix of percentage contributions

    References:
        Yamai & Yoshiba (2002). Comparative analyses of expected shortfall
        and value-at-risk.

    Examples:
        >>> model = fit_tsfm(data, asset_names=["A1"], factor_names=["F1", "F2"])
        >>> es_decomp = fm_es_decomp(model, p=0.05)
        >>> es_decomp['es_fm']  # 5% ES (CVaR)
        array([-0.0312])
        >>> es_decomp['pc_es']  # Percentage contributions
        array([[54.1, 33.2, 12.7]])
    """
    N = len(model.asset_names)
    K = len(model.factor_names)

    # Get beta_star: N x (K+1)
    beta = model.beta.copy()
    beta[np.isnan(beta)] = 0
    beta_star = np.column_stack([beta, model.resid_sd])

    # Extract factor returns and compute standardized residuals
    factor_df = model.data.select(model.factor_names).to_pandas()
    factors = factor_df.values

    # Standardized residuals
    resid_std = model.residuals / model.resid_sd[:, np.newaxis]

    # Initialize arrays
    var_fm = np.full(N, np.nan)
    es_fm = np.full(N, np.nan)
    m_es = np.full((N, K + 1), np.nan)
    c_es = np.full((N, K + 1), np.nan)
    pc_es = np.full((N, K + 1), np.nan)

    if type == "normal":
        # Get factor covariance
        if factor_cov is None:
            if use == "pairwise":
                factor_cov = factor_df.cov().values
            else:
                factor_df_complete = factor_df.dropna()
                factor_cov = factor_df_complete.cov().values

        # Build Cov(F_star)
        factor_star_cov = np.eye(K + 1)
        factor_star_cov[:K, :K] = factor_cov

        # Factor expected returns
        MU = np.concatenate([factor_df.mean().values, [0]])

        # SIGMA * Beta
        SIGB = beta_star @ factor_star_cov

    # Loop over assets
    for i in range(N):
        # Get asset returns
        asset_name = model.asset_names[i]
        R = model.data[asset_name].to_numpy()

        # Remove NaNs
        valid_idx = ~np.isnan(R)
        R_valid = R[valid_idx]

        if len(R_valid) == 0:
            continue

        if type == "np":
            # Non-parametric VaR
            var_fm[i] = np.quantile(R_valid, p)

            # Find exceedances (R <= VaR)
            exceed_mask = R_valid <= var_fm[i]
            exceed_idx = np.where(exceed_mask)[0]

            if len(exceed_idx) == 0:
                continue

            # ES: mean of returns in tail
            es_fm[i] = np.mean(R_valid[exceed_idx])

            # Get F_star
            factors_valid = factors[valid_idx, :]
            resid_std_valid = resid_std[i, valid_idx]
            factor_star = np.column_stack([factors_valid, resid_std_valid])

            # Marginal ES: mean of factor returns in tail
            m_es[i, :] = np.mean(factor_star[exceed_idx, :], axis=0)

        elif type == "normal":
            # Normal ES
            sd_i = np.sqrt(beta_star[i, :] @ factor_star_cov @ beta_star[i, :])
            es_fm[i] = -(
                beta_star[i, :] @ MU +
                sd_i * stats.norm.pdf(stats.norm.ppf(p)) / p
            )

            # Marginal ES for normal
            m_es[i, :] = -(
                MU + SIGB[i, :] / np.std(R_valid, ddof=1) *
                stats.norm.pdf(stats.norm.ppf(p)) / p
            )

        # Correction factor to ensure sum(cES) = ES
        component_sum = np.sum(m_es[i, :] * beta_star[i, :])
        if component_sum != 0:
            cf = es_fm[i] / component_sum
        else:
            cf = 1.0

        # Apply correction and compute components
        m_es[i, :] = cf * m_es[i, :]
        c_es[i, :] = m_es[i, :] * beta_star[i, :]
        pc_es[i, :] = 100 * c_es[i, :] / es_fm[i]

    return {
        "es_fm": es_fm,
        "m_es": m_es,
        "c_es": c_es,
        "pc_es": pc_es
    }
