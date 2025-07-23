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

``` text
mai25-bds-trafic-cycliste/
├── app/                    # Streamlit app
│   ├── main.py             # main entry point of the app
│   ├── config.py
│   ├── utils/
│   │   ├── model_logic.py
│   │   └── streamlit_helpers.py
│   └── sections/
│       ├── home.py
│       ├── data_visualization.py
│       ├── data_exploration.py
│       └── modeling.py
├── smartcheck/             # Project Core logic
│   ├── logger_config.py                    # Log management tools
│   ├── dataframe_common.py                 # Shared data loading tools
│   ├── classification_common.py            # Classification Modeling tools
│   ├── meta_search_common.py               # Multi-strategy hyperparameter tuning tools
│   ├── dataframe_project_specific.py       # Advanced project specific feature engineering tools
│   ├── preprocessing_project_specific.py   # Advanced project specific transformers
│   └── resources/
│       └── config.yaml
├── tests/                  # Unit tests (pytest for core and app)
├── notebooks/              # Jupyter notebooks (not packaged)
├── README.md               # Project documentation
├── LICENSE                 # MIT license
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
# Create virtual environment using NOX and Conda
conda activate base
pip install --upgrade pip
pip install nox
```

> ⚠️ At this point, `nox`, `pre-commit`, and `nbstripout` are not yet available. Install the project (next section) before activating hooks.

---

### 🚀 Day-to-day Usage

```bash
# Rebuild and complete virtual env for standard streamlit application and notebooks (+ trigger test/flake8)
nox -s build-3.12 --reuse-existing

# Rebuild and complete virtual env for pytorch deep learning notebooks
nox -s dl-torch-3.12 --reuse-existing

# Rebuild and complete virtual env for tensorflow deep learning notebooks (restricted to python 3.9)
nox -s dl-tensorflow-3.9 --reuse-existing

# Activate the conda env listed via:
conda env list
conda activate [env_path]

# Optional: clean (project generated file only)
nox -s clean_project

# Optional: clean everything (/!\ including virtual environment generated with conda/nox)
nox -s clean_all

# Optional: trigger packaging construction
nox -s package
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
streamlit run app/main.py
```

You can navigate between pages from the sidebar:

- 🏠 Introduction au projet
- ⚙️ Démarche projet et résultats
- 🔍 Exploration des données
- 📈 Visualisation et Statistiques
- 🧪 Évaluation des modèles

---

## 📓 Notebooks

The `notebooks/` folder contains various exploratory notebooks (school and project related), showcasing:

- 🧼 Data cleaning and preprocessing strategies  
- 📊 Exploratory Data Analysis with Matplotlib, Plotly, Seaborn, and Bokeh  
- 🎯 Resampling methods and hyperparameter tuning (GridSearch, RandomizedSearch, BayesSearch)  
- 🧠 Training and evaluation of baseline and advanced ML models  
- 🤖 Neural network experimentation with tensorflow and pytorch  

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
