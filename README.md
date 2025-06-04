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

### 3. ⚙️ MLOps
- Code packaging and automation
- Reproducibility and continuous testing

---

## 🧱 Project Structure

```
mai25-bds-trafic-cycliste/
├── smartcheck/         # Source code (project core)
│   ├── [modules].py
│   └── resources/
│       └── config.yaml
├── tests/              # Unit tests (pytest)
├── notebooks/          # Jupyter notebooks (not packaged)
├── README.md           # Project documentation
├── LICENSE             # MIT license
├── requirements.txt    # Pip requirements
├── MANIFEST.in         # Packaging resources configuration for setuptools
├── pyproject.toml      # Python project configuration
├── noxfile.py          # NOX session configuration
└── .coveragerc         # Test coverage configuration
```

---

## ⚙️ Installation

### Option 1: Using NOX/CONDA (recommended for multi-environment workflows)

```bash
# Prerequisites
conda activate base
python -m pip install --upgrade pip
pip install nox

# Virtual envs creation in .nox and activation
nox -s
conda env list # check the build env location
conda activate [buildEnvLocation]

# (Re)Install development dependencies / recheck code rules / retest with coverage
nox -s build-3.12 --reuse-existing

# Clean environments and project files
conda deactivate
nox -s clean_all
nox -s clean_project

# Build final package
nox -s package

# List the nox session available (noxfile.py)
nox --list
```

### Option 2: Using native python and its native virtual environment

```bash
python -m pip install --upgrade pip

# Virtual env création in and activation
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# (Re)Install development dependencies / recheck code rules / retest with coverage
pip install -e .[dev]
flake8
pytest

# Build final package
python -m build
```

---

## 🚀 Usage

You can import the module in Python:

```python
from smartcheck.dataframe_common import load_dataset_from_config
df = load_dataset_from_config("velib_dispo_data", sep=";")
```

Or explore notebooks in the `notebooks/` directory.

---

## 🔧 Configuration

All configuration settings are in:

```
smartcheck/resources/config.yaml
```

Access safely with:

```python
import importlib.resources
with importlib.resources.files("smartcheck.resources").joinpath("config.yaml").open("r") as f:
    ...
```

---

## 🔄 Continuous Integration

GitHub Actions executes all tests and tracks code coverage:

- `ci_main.yml`: for main branch activity (push, PR)
- `ci_branch.yml`: for all other branches

Coverage data is sent to [Codecov](https://codecov.io/gh/zheddhe/mai25-bds-trafic-cycliste).

---

## 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi  
- Raphaël Parmentier