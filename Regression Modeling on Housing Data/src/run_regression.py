from __future__ import annotations
import argparse
import numpy as np

from .utils import Paths, set_seed, save_json, plot_bars
from .data import load_regression
from .models import regression_models
from .train_eval import evaluate_regression, rank_models_by_r2

def main():
    parser = argparse.ArgumentParser(description="Run California Housing regression experiments")
    parser.add_argument("--all", action="store_true", help="run all predefined models (default)")
    parser.add_argument("--models", nargs="*", default=None, help="optional subset e.g. ridge gboost random_forest")
    args = parser.parse_args()

    paths = Paths.from_root(".")
    set_seed(42)

    # Load data
    ds = load_regression()

    # Select models
    reg_models = regression_models()
    if args.models:
        missing = [m for m in args.models if m not in reg_models]
        if missing:
            print("Unknown model(s):", ", ".join(missing))
            print("Available:", ", ".join(reg_models.keys()))
            return
        reg_models = {k: reg_models[k] for k in args.models}

    # Train & evaluate
    results = {}
    for name, model in reg_models.items():
        model.fit(ds.X_train, ds.y_train)
        y_pred = model.predict(ds.X_test)
        rep = evaluate_regression(ds.y_test, y_pred)
        results[name] = rep.__dict__

        # Save model-specific figures where meaningful
        # 1) Coefficients for linear models
        if hasattr(model, "coef_"):
            try:
                coefs = np.asarray(model.coef_).ravel()
                plot_bars(
                    coefs,
                    ds.feature_names,
                    out_path=str(paths.figures / f"feature_importance_{name}.png"),
                    title=f"{name} coefficients (standardized features)"
                )
            except Exception:
                pass

        # 2) Feature importance for tree ensembles
        if hasattr(model, "feature_importances_"):
            try:
                plot_bars(
                    model.feature_importances_,
                    ds.feature_names,
                    out_path=str(paths.figures / f"feature_importance_{name}.png"),
                    title=f"{name} feature importance"
                )
            except Exception:
                pass

    # Persist results
    out_path = paths.results / "regression_results.json"
    save_json(out_path, results)

    # Print concise summary
    print("Saved results ->", out_path)
    for k, v in results.items():
        print(f"{k:16s} | MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}  R2={v['r2']:.3f}")

    ranked = rank_models_by_r2(results)
    print("\nBest by R²:")
    for name, r2 in ranked:
        print(f"  {name:16s} R2={r2:.3f}")

if __name__ == "__main__":
    main()
