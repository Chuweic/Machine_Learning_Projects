from __future__ import annotations
from typing import Dict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def regression_models() -> Dict[str, object]:
    """A small but diverse set of regressors."""
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.001, max_iter=10000, random_state=42),
        "svr_rbf": SVR(kernel="rbf", C=10, gamma="scale"),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "gboost": GradientBoostingRegressor(random_state=42),
        "mlp_reg": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42),
    }
