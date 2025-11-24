# Regression Benchmark Report — California Housing

## Goal
Predict median house value (`MedHouseVal`) from 8 standardized socioeconomic/geographic features.

## Models Compared
Linear Regression, Ridge, Lasso, SVR (RBF), Random Forest, Gradient Boosting, MLP Regressor.


## Results (Test Set)

| Model | MAE (↓) | RMSE (↓) | R² (↑) |
|---|---:|---:|---:|
| random_forest | 0.326 | 0.503 | 0.807 |
| mlp_reg | 0.343 | 0.513 | 0.799 |
| gboost | 0.372 | 0.542 | 0.776 |
| svr_rbf | 0.377 | 0.569 | 0.753 |
| lasso | 0.533 | 0.745 | 0.577 |
| ridge | 0.533 | 0.746 | 0.576 |
| linear_regression | 0.533 | 0.746 | 0.576 |

## Key Takeaways
- Best model by R²: **random_forest**
- Ensemble methods (Random Forest / Gradient Boosting) generally outperform linear baselines on nonlinear tabular patterns.
- `MedInc` is typically the strongest feature signal for price prediction.

## Artifacts
- Metrics: `reports/results/regression_results.json`
- Feature plots: `reports/figures/feature_importance_*.png`
