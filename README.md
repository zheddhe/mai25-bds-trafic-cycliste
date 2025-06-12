# 🚲 Cyclist Traffic ML Project

[![codecov](https://codecov.io/gh/zheddhe/mai25-bds-trafic-cycliste/graph/badge.svg?token=6TLD3FM08Z)](https://codecov.io/gh/zheddhe/mai25-bds-trafic-cycliste)
[![CI Main](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions)
[![CI Branch](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions)

> A machine learning pipeline for analyzing bike traffic data in Paris.  
> Developed as part of the April 2025 Machine Learning Engineering (MLE) full training program.

---

## 🧭 Overview

This project implements a full machine learning and MLOps pipeline in three main stages:

### 1. 📐 Data Product Management
- Define business goals
- Scope the data lifecycle

### 2. 📊 Data Science
- Data analysis and visualization
- Model development and evaluation
- Advanced preprocessing helpers and strategies

### 3. ⚙️ MLOps
- Code packaging and automation
- Reproducibility and continuous testing

---

## 🧱 Project Structure

```
mai25-bds-trafic-cycliste/
├── smartcheck/             # Source code (project core)
│   ├── dataframe_common.py         # Shared data loading and logging functions
│   ├── classification_common.py    # Resampling and threshold optimization tools
│   ├── meta_search_common.py       # Multi-strategy hyperparameter tuning
│   ├── dataframe_project_specific.py # Geo and datetime transformations
│   └── resources/
│       └── config.yaml
├── tests/                  # Unit tests (pytest)
├── notebooks/              # Jupyter notebooks (not packaged)
├── README.md               # Project documentation
├── LICENSE                 # MIT license
├── requirements.txt        # Pip requirements
├── MANIFEST.in             # Packaging resources configuration for setuptools
├── pyproject.toml          # Python project configuration
├── noxfile.py              # NOX session configuration
├── .pre-commit-config.yaml # Pre-commit configuration (clean jupyter notebooks before commit)
└── .coveragerc             # Test coverage configuration
```

---

## ⚙️ Installation

### 🔧 Initial Setup (One-time)

```bash
# Create virtual environment (preferred: NOX or fallback: venv+pip)

# Option 1: Using NOX + Conda (recommended)
conda activate base
pip install --upgrade pip
pip install nox

# Option 2: Using native Python
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
```

> ⚠️ At this point, `nox`, `pre-commit`, and `nbstripout` are not yet available. Install the project (next section) before activating hooks.

---

### 🚀 Day-to-day Usage

#### Option A: With NOX (recommended)

```bash
# Create environment and install dev deps
nox -s build-3.12

# Activate the conda env listed via:
conda env list
conda activate [env_path]

# Optional: clean or rebuild
nox -s clean_all
nox -s package
```

#### Option B: With pip only

```bash
# Install development dependencies
pip install -e .[dev]

# Run checks
flake8
pytest
```

---

### 🪝 Activate Commit Hooks (after environment is built)

```bash
# Activate pre-commit hooks (mandatory)
pre-commit install

# (Optional) Activate strip out of files when stagging
nbstripout --install

# (optional Deactivate strip out of files when stagging
nbstripout --install
```

---

## 📘 Usage Examples

In notebooks or scripts:

```python
# Load datasets (local or Google Drive)
from smartcheck.dataframe_common import load_dataset_from_config
df = load_dataset_from_config("velib_dispo_data", sep=";")

# View structure and missing data summary
from smartcheck.dataframe_common import log_general_info
log_general_info(df)
```

Additional helpers include:

- `detect_and_log_duplicates_and_missing(df)`
- `normalize_column_names(df)`
- `analyze_by_reference_variable(df, "some_col")`
- `cross_validation_with_resampling(X, y, model)`
- `cross_validation_with_resampling_and_threshold(X, y, model, thresholds=np.arange(...))`
- `compare_search_methods(...)` for GridSearchCV / RandomizedSearchCV / BayesSearchCV

---

## 🔧 Configuration

All configuration settings are in:

```
smartcheck/resources/config.yaml
```

Example access:

```python
from smartcheck.paths import load_config
config = load_config()
print(config["data"]["input"]["velib_dispo_data"])
```

---

## 🔄 Continuous Integration

GitHub Actions executes all tests and tracks code coverage:

- `ci_main.yml`: "main" branch
- `ci_branch.yml`: "other than main" branches

Coverage is sent to [Codecov](https://codecov.io/gh/zheddhe/mai25-bds-trafic-cycliste).

---

## 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi  
- Raphaël Parmentier
