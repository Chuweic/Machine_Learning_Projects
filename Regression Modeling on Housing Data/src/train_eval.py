from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass
class RegressionReport:
    mae: float
    rmse: float
    r2: float

def evaluate_regression(y_true, y_pred) -> RegressionReport:
    return RegressionReport(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )

def rank_models_by_r2(results: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
    return sorted(((k, v["r2"]) for k, v in results.items()),
                  key=lambda kv: kv[1], reverse=True)
