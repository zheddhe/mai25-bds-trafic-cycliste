# `avr25-mle-velib` 🚲

[![codecov](https://codecov.io/gh/zheddhe/avr25-mle-velib/graph/badge.svg?token=6TLD3FM08Z)](https://codecov.io/gh/zheddhe/avr25-mle-velib)
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
├── smartcheck/                # Module source code
│   ├── __init__.py
│   ├── [modules].py
│   ├── ...
│   └── resources/
│       └── config.yaml
│
├── tests/                     # Module Unit tests (with pytest)
│   ├── test_[modules].py
│   └── ...
│
├── notebooks/                 # Module Jupyter Notebooks (not packaged in builds)
│   ├── [notebook].ipynb
│   └── ...
│
├── README.md                  # Project doc (this file)
├── LICENSE                    # MIT license file
├── requirements.txt           # basic module tracking (used with pip)
├── MANIFEST.in                # Package resource description (for setuptools used by pyproject)
├── pyproject.toml             # python project configuration file (for python and pip)
├── noxfile.py                 # NOX virtual env setup and session
└── .coveragerc                # Test Coverage specific settings
```

---

## 🛠️ Installation

### 📦 Cloner le dépôt
```bash
git clone https://github.com/zheddhe/avr25-mle-velib.git
cd avr25-mle-velib
```

---

### ⚙️ Option 1 : Avec [NOX](https://nox.thea.codes/) (gestion multi-environnements virtuels)
```bash
# Prérequis : Assurez-vous d’avoir Python et pip à jour
python -m pip install --upgrade pip
pip install nox

# Installation + Build + Tests avec couverture
nox -s full --reuse-existing

# Nettoyage complet (y compris les environnements .nox/*)
nox -s clean_all

# Nettoyage partiel du projet (sans supprimer les environnements .nox/*)
nox -s clean_project
```

---

### ⚙️ Option 2 : Avec un environnement virtuel classique (.venv)
```bash
# Prérequis : Assurez-vous d’avoir Python et pip à jour
python -m pip install --upgrade pip

# Créer et activer un environnement virtuel (exemple avec venv)
python -m venv .venv
source .venv/bin/activate  # sous Linux/macOS
# .venv\Scripts\activate    # sous Windows

# Installation en mode développement
pip install -e .[dev]

# Lancer les tests avec couverture
pytest --cov=src tests/
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

Run all tests with coverage:

```bash
pytest
```

Open `htmlcov/index.html` for a visual local report.

---

## 🔄 Continuous Integration

- GitHub Actions runs all tests and reports coverage.
-- On main branch for push and pull requests with configuration file: `.github/workflows/ci_main.yml` 
-- On issue branches with configuration file `.github/workflows/ci_branch.yml`
- Coverage report is sent to Codecov (badge above) for main branch activity only.

---

## 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi  
- Raphaël Parmentier
