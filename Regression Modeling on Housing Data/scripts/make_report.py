import json
from pathlib import Path

RESULTS_PATH = Path("reports/results/regression_results.json")
REPORT_PATH = Path("REPORT.md")

def main():
    results = json.loads(RESULTS_PATH.read_text())

    # sort by r2 desc
    ranked = sorted(results.items(), key=lambda kv: kv[1]["r2"], reverse=True)
    best_model = ranked[0][0]

    rows = []
    for name, m in ranked:
        rows.append(
            f"| {name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.3f} |"
        )

    table = "\n".join(rows)

    md = f"""# Regression Benchmark Report — California Housing

## Goal
Predict median house value (`MedHouseVal`) from 8 standardized socioeconomic/geographic features.

## Models Compared
Linear Regression, Ridge, Lasso, SVR (RBF), Random Forest, Gradient Boosting, MLP Regressor.

## Results (Test Set)

| Model | MAE (↓) | RMSE (↓) | R² (↑) |
|---|---:|---:|---:|
{table}

## Key Takeaways
- Best model by R²: **{best_model}**
- Ensemble methods (Random Forest / Gradient Boosting) generally outperform linear baselines on nonlinear tabular patterns.
- `MedInc` is typically the strongest feature signal for price prediction.

## Artifacts
- Metrics: `reports/results/regression_results.json`
- Feature plots: `reports/figures/feature_importance_*.png`
"""
    REPORT_PATH.write_text(md)
    print("Wrote REPORT.md")

if __name__ == "__main__":
    main()
