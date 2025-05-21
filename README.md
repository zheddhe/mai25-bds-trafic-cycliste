# Project `avr25-mle-velib`

## 🗂️ Project Overview

This project is structured around a three-phase machine learning pipeline:

- **Phase 1 – Data Product Management**  
  Defining business goals, project scoping, and data strategy.
  
- **Phase 2 – Data Science**  
  Data exploration, cleaning, modeling, and evaluation.
  
- **Phase 3 – MLOps**  
  Packaging, automation, configuration management, and reproducibility.

---

## 🏗️ Project Structure

```text
avr25-mle-velib/
├── README.md              # Project documentation (this file)
├── requirements.txt       # Python dependencies (for dev with pip install -r requirements.txt)
├── pyproject.toml         # Meta information and dependencies (for build with pip install .)
│
├── config/                # Static configuration files (YAML, JSON, etc.)
│   └── config.yaml
│
├── my_project/            # Main source code (Python modules and logic)
│   ├── __init__.py
│   ├── ...
│
├── tests/                 # Unit and integration tests
│   ├── test_core.py
│   ├── ...
│
└── notebooks/             # Jupyter notebooks for exploratory analysis
    ├── 00_exploratory_analysis.ipynb
    ├── ...

