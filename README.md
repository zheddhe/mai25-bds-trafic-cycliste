# `avr25-mle-velib` 🚲

[![codecov](https://codecov.io/gh/zheddhe/avr25-mle-velib/branch/main/graph/badge.svg)](https://codecov.io/gh/zheddhe/avr25-mle-velib)
> [![CI](https://github.com/zheddhe/avr25-mle-velib/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-velib/actions)  
> [![CI](https://github.com/zheddhe/avr25-mle-velib/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-velib/actions)  
> 📦 A machine learning pipeline for analyzing Velib data  
> Developed during the April 2025 MLE training program.

---

## 🧠 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Continuous Integration](#-continuous-integration)
- [Tooling](#-tooling)
- [License](#-license)
- [Contributors](#-contributors)

---

## 🧠 Overview

This project implements a 3-phase machine learning and MLOps pipeline:

### 📐 Phase 1 – Data Product Management
- Define business goals
- Scope the data lifecycle

### 📊 Phase 2 – Data Science
- Data analysis & visualization
- Modeling and evaluation

### ⚙️ Phase 3 – MLOps
- Code packaging and automation
- Reproducibility and testing

---

## 🏗️ Project Structure

```text
avr25-mle-velib/
├── smartcheck/                # Core source code
│   ├── __init__.py
│   ├── dataframe_common.py
│   ├── ...
│   └── resources/
│       └── config.yaml
│
├── tests/                     # Unit tests (pytest)
│   ├── test_dataframe_common.py
│   └── ...
│
├── notebooks/                 # Jupyter Notebooks
│   ├── 00_exploratory_analysis.ipynb
│   └── ...
│
├── pyproject.toml             # Build config
├── requirements.txt           # Alternative dev install
├── README.md                  # Project doc (this file)
├── .coveragerc                # Coverage settings
├── MANIFEST.in                # Package data inclusion
└── .github/workflows/test.yml # CI config
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/zheddhe/avr25-mle-velib.git
cd avr25-mle-velib

# Install in editable mode with dev tools
pip install -e .[dev]
```

---

## 🚀 Usage

You can import and use the main parts as a Python module:

```python
from smartcheck.dataframe_common import load_dataset_from_config

df = load_dataset_from_config("velib_dispo_data", sep=";")
```

You can also run experiments or analyses directly in Jupyter Notebooks.

---

## ⚙️ Configuration

Configuration is loaded from a YAML file located at:

```
smartcheck/resources/config.yaml
```

Use `importlib.resources` to access it safely:

```python
import importlib.resources
with importlib.resources.files("smartcheck.resources").joinpath("config.yaml").open("r") as f:
    ...
```

---

## ✅ Testing

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=smartcheck --cov-report=term --cov-report=html
```

Open `htmlcov/index.html` for a visual report.

---

## 🔄 Continuous Integration

- GitHub Actions runs all tests and reports coverage on every push/pull request.
- Configuration file: `.github/workflows/test.yml`
<!-- TODO: add a codecov account and report it
- Coverage report is sent to Codecov (badge above).
-->

---

## 🧰 Tooling

| Tool               | Purpose                                    |
|--------------------|--------------------------------------------|
| `pytest`           | Test runner                                |
| `pytest-cov`       | Test coverage reporting                    |
| `importlib.resources` | Access bundled config files safely     |

---

## 📄 License

MIT License © 2025  
Developed as part of the MLE April 2025 promotion.

---

## 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi  
- Raphaël Parmentier
