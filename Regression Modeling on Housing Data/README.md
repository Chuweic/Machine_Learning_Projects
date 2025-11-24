# Project 1 — California Housing Regression (ML Intern Portfolio)

Predict median house value using classic ML regression models. Compares
Linear/Ridge/Lasso, SVR, RandomForest, GradientBoosting, and MLPRegressor.

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

ml-regression-housing/
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ src/
   ├─ __init__.py
   ├─ data.py
   ├─ models.py
   ├─ train_eval.py
   ├─ utils.py
   └─ run_regression.py
