# Project 1 — California Housing Regression (ML Intern Portfolio)

Predict median house value using classic ML regression models. Compares
Linear/Ridge/Lasso, SVR, RandomForest, GradientBoosting, and MLPRegressor.


## Project Structure
    ml-regression-housing/
    ├─ README.md
    ├─ REPORT.md                 # Automatically generated model report
    ├─ requirements.txt
    ├─ .gitignore
    ├─ script/
    │   └─ scriptmake_report.py  # Script to auto-build REPORT.md
    └─ src/
        ├─ __init__.py
        ├─ data.py               # Dataset loader + preprocessing
        ├─ models.py             # Model registry
        ├─ train_eval.py         # Metric functions + evaluation
        ├─ utils.py              # Utility helpers (paths, plotting, seed)
        └─ run_regression.py     # Main experiment runner



    Output files:
    reports/
    ├─ results/
    │    └─ regression_results.json      # model performance metrics
    └─ figures/
        └─ feature_importance_*.png     # optional coefficient / importance plots


## Quickstart

```bash
# (Recommended) create & activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all regression experiments and generate outputs in reports/
python -m src.run_regression --all
