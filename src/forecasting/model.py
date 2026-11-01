"""XGBoost battery degradation (capacity fade) forecaster.

Target: capacity_fade_pct, the percentage of rated capacity lost since a
battery's first discharge cycle. This is the standard target for this
dataset in published State-of-Health/Remaining-Useful-Life work, it maps
directly onto NASA's own end-of-life criterion (30% fade).

Evaluation is a group holdout by battery_id, not a random row split. A
random split would let later cycles of a battery the model has already
partially seen leak into training, which inflates reported accuracy and
doesn't reflect the real deployment scenario: forecasting degradation on a
battery/vehicle the model has never seen telemetry from before.

Note on regime_cluster: all four batteries in this dataset (B0005/6/7/18)
were run at a constant 24C ambient temperature, so the hot/cold/mild bucket
built in extract_battery_features.py is a zero-variance feature here and is
deliberately excluded from the model. battery_id is used as the grouping/
regime-equivalent signal instead, since that's where the actual behavioral
diversity in this dataset lives (NASA varied discharge protocols per
battery, not temperature, across this particular quartet).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "battery_degradation.csv"

FEATURE_COLUMNS = [
    "discharge_cycle_index",
    "mean_voltage_v",
    "min_voltage_v",
    "mean_current_a",
    "mean_temperature_c",
    "max_temperature_c",
    "discharge_duration_s",
]
TARGET_COLUMN = "capacity_fade_pct"
HOLDOUT_BATTERY = "B0018"  # smallest battery in the set; commonly held out in published work on this dataset


@dataclass
class EvalMetrics:
    mae: float
    rmse: float
    r2: float


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def group_holdout_split(df: pd.DataFrame, holdout_battery: str = HOLDOUT_BATTERY):
    train_df = df[df["battery_id"] != holdout_battery]
    test_df = df[df["battery_id"] == holdout_battery]
    return (
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        test_df[FEATURE_COLUMNS],
        test_df[TARGET_COLUMN],
    )


def evaluate(y_true, y_pred) -> EvalMetrics:
    return EvalMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2=float(r2_score(y_true, y_pred)),
    )


def train(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    experiment_name: str = "battery_degradation_forecaster",
) -> tuple[xgb.XGBRegressor, EvalMetrics]:
    df = load_data()
    X_train, y_train, X_test, y_test = group_holdout_split(df)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        mlflow.log_params(params)
        mlflow.log_param("holdout_battery", HOLDOUT_BATTERY)
        mlflow.log_param("train_batteries", sorted(df[df["battery_id"] != HOLDOUT_BATTERY]["battery_id"].unique().tolist()))
        mlflow.log_param("features", FEATURE_COLUMNS)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        mlflow.log_metric("mae", metrics.mae)
        mlflow.log_metric("rmse", metrics.rmse)
        mlflow.log_metric("r2", metrics.r2)

        for name, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
            mlflow.log_metric(f"feature_importance_{name}", float(importance))

        mlflow.xgboost.log_model(model, artifact_path="model")

    return model, metrics


if __name__ == "__main__":
    model, metrics = train()
    print(f"Holdout battery: {HOLDOUT_BATTERY}")
    print(f"MAE:  {metrics.mae:.3f} percentage points of capacity fade")
    print(f"RMSE: {metrics.rmse:.3f}")
    print(f"R2:   {metrics.r2:.3f}")
