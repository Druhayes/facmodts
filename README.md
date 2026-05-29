# facmodts_py: Time Series Factor Models in Python

Python implementation of the facmodTS R package for fitting time series factor models using robust methods. Companion to *Robust Portfolio Construction and Risk Analysis* (Springer, 2025).

## Implementation Status

**Current Phase**: Phase 1 - Foundation & Core Fitting (In Progress)

### Phase 1: Foundation & Core Fitting ✅ (Partial)

**Completed**:
- ✅ Data models (TsfmModel, TsfmControl, TsfmUpDn)
- ✅ Control parameter handling (fit_tsfm_control)
- ✅ Utility functions (DLS weights, excess returns, NA handling)
- ✅ LS (Ordinary Least Squares) regression
- ✅ DLS (Discounted Least Squares) regression
- ✅ Test infrastructure (fixtures, tolerances, synthetic data)
- ✅ Basic unit tests (25+ tests)

**In Progress**:
- R comparison tests
- Additional edge case tests

**Remaining**:
- Comprehensive documentation
- Performance benchmarking

### Phase 2: Robust Regression & Variable Selection (Planned)

- Robust regression (M-estimators via statsmodels RLM)
- Stepwise variable selection
- Best subsets variable selection
- LARS/Lasso variable selection

### Phase 3: Risk Decomposition (Planned)

- Factor model covariance (fm_cov)
- SD decomposition (fm_sd_decomp)
- VaR decomposition (fm_var_decomp)
- ES decomposition (fm_es_decomp)

### Phase 4: Wrapper Functions (Planned)

- Market-timing models (fit_tsfm_mt)
- Up/Down market models (fit_tsfm_up_dn)
- Lag/Lead beta models
- Fama-French wrappers (FF3, FF4)

### Phase 5: Performance Attribution (Planned)

- Performance attribution (pa_fm)
- Summary methods
- Print methods
- Predict methods

### Phase 6: Visualization (Planned)

- Comprehensive plotting suite (12+ plot types)
- Matplotlib and Plotly support

## Installation

```bash
# Using UV (recommended)
uv pip install -e ".[test]"

# Or using pip
pip install -e ".[test]"
```

## Quick Start

```python
import polars as pl
from facmodts import fit_tsfm

# Load your data (date, asset returns, factor returns)
data = pl.read_csv("returns.csv")

# Fit LS model
model = fit_tsfm(
    data=data,
    asset_names=["Asset1", "Asset2", "Asset3"],
    factor_names=["Factor1", "Factor2", "Factor3"],
    fit_method="LS"
)

# Extract results
print("Alpha (intercepts):", model.alpha)
print("Beta (factor loadings):\n", model.beta)
print("R-squared:", model.r_squared)
print("Residual SD:", model.resid_sd)
```

### DLS (Discounted Least Squares)

```python
# DLS gives more weight to recent observations
model_dls = fit_tsfm(
    data=data,
    asset_names=["Asset1", "Asset2"],
    factor_names=["Factor1", "Factor2"],
    fit_method="DLS",
    decay=0.95  # Default decay factor
)
```

### Excess Returns

```python
# Subtract risk-free rate from returns
model_excess = fit_tsfm(
    data=data,
    asset_names=["Asset1", "Asset2"],
    factor_names=["MKT", "SMB", "HML"],
    rf_name="RF",  # Risk-free rate column
    fit_method="LS"
)
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=facmodts --cov-report=term-missing

# Run specific test file
pytest tests/test_fitting.py

# Run tests that don't require R
pytest -m "not r_comparison"
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev,test,plot]"

# Format code
black facmodts/ tests/

# Lint code
ruff check facmodts/ tests/

# Type checking
mypy facmodts/
```

## Project Structure

```
facmodts_py/
├── facmodts/               # Main package
│   ├── __init__.py
│   ├── models.py          # TsfmModel, TsfmControl data classes
│   ├── fitting.py         # fit_tsfm() and helpers
│   ├── control.py         # fit_tsfm_control()
│   ├── utils.py           # Utility functions (Numba-accelerated)
│   └── ...                # (Phase 2-6 modules)
│
├── tests/
│   ├── conftest.py        # Fixtures, tolerances
│   ├── test_fitting.py    # Core fitting tests
│   ├── data/
│   │   └── stocks_factors.csv  # Real test data (294 assets)
│   └── ...
│
├── examples/              # Usage examples
├── docs/                  # Documentation
├── pyproject.toml        # Package configuration
└── README.md
```

## Key Differences from R facmodTS

1. **Data Handling**: Uses Polars for data operations (2-5x faster), converts to pandas for statsmodels regression
2. **Robust Methods**: Python uses M-estimators (Huber, Bisquare) via statsmodels RLM, whereas R uses MM-estimators. Breakdown points may differ.
3. **Column Names**: Automatically converts spaces to periods like R xts
4. **NA Handling**: Per-asset NA removal to allow unequal histories (same as R)

## License

GPL-3.0 (same as R package)

## Authors

Python conversion team, based on the original facmodTS R package.

## References

*Robust Portfolio Construction and Risk Analysis* (2025). Springer.
