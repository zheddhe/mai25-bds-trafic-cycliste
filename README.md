# 🚲 Cyclist Traffic ML Project

[![codecov](https://codecov.io/gh/zheddhe/mai25-bds-trafic-cycliste/graph/badge.svg?token=6TLD3FM08Z)](https://codecov.io/gh/zheddhe/mai25-bds-trafic-cycliste)
[![CI Main](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions)
[![CI Branch](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/mai25-bds-trafic-cycliste/actions)

> A machine  learning pipeline for analyzing bike traffic data in Paris.  
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
├── app/                    # Streamlit app
│   ├── home.py
│   ├── data_visualization.py
│   ├── data_exploration.py
│   ├── modeling.py
│   ├── utils/
│   └── config.py
├── smartcheck/             # Source code (project core)
│   ├── logger_config.py                    # Log management tools
│   ├── dataframe_common.py                 # Shared data loading tools
│   ├── classification_common.py            # Classification Modeling tools
│   ├── meta_search_common.py               # Multi-strategy hyperparameter tuning tools
│   ├── dataframe_project_specific.py       # Advanced project specific feature engineering tools
│   ├── preprocessing_project_specific.py   # Advanced project specific transformers
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

# Optional: clean (project generated file only)
nox -s clean_project

# Optional: clean everything (/!\ including virtual environment generated with conda/nox)
nox -s clean_all

# Optional: trigger packaging construction
nox -s package

# Optional: set up deep learning specific env (python 3.9)
nox -s dl-3.9
```

#### Option B: With pip only

```bash
# Install development dependencies
pip install -e .[py312, dev]

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
nbstripout --uninstall
```

---

## 🚀 Streamlit App (Interactive Demo)

The project includes a full **Streamlit web application** located in `app/`.  
This interactive app allows for data exploration, visualization, and model result inspection.

### ▶️ Launch the app

```bash
streamlit run streamlit_app.py
```

You can navigate between pages from the sidebar:
- 🏠 Home
- 🔍 Data Exploration
- 📊 Data Visualization
- 🧪 Model Evaluation

---

## 📓 Notebooks

The `notebooks/` folder contains various exploratory notebooks (school and project related), showcasing:

- 🧼 Data cleaning and preprocessing strategies  
- 📊 Exploratory Data Analysis with Matplotlib, Plotly, Seaborn, and Bokeh  
- 🎯 Resampling methods and hyperparameter tuning (GridSearch, RandomizedSearch, BayesSearch)  
- 🧠 Training and evaluation of baseline and advanced ML models  
- 🤖 Neural network experimentation with Keras  

All notebooks leverage reusable functions from the `smartcheck` module.

---

## 🧪 Testing and Continuous Integration

Tests are executed using `pytest`, including:

- ✅ Smoke tests and logic mocks for the Streamlit app  
- ✅ Unit tests for core reusable modules (`smartcheck/`)  

CI workflows are handled by GitHub Actions:

- `ci_main.yml`: runs on every push to the `main` branch  
- `ci_branch.yml`: runs on all feature and non-main branches  

📈 Code coverage results are automatically uploaded to [Codecov](https://codecov.io/gh/zheddhe/mai25-bds-trafic-cycliste) after tests on the `main` branch.

---

## 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi  
- Raphaël Parmentier
