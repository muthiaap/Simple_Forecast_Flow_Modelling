import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
import matplotlib.pyplot as plt
import pandas as pd

def setup_mlflow(experiment_name="stock_forecasting"):
    mlflow.set_experiment(experiment_name)
    print(f"MLflow initialized with experiment: {experiment_name}")

def log_metrics_params_model(model, params: dict, metrics: dict,
                             model_name: str = "xgboost_model",
                             run_name: str | None = None):

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if model_name == "xgboost_model":
            # nama artifact_path tidak boleh mengandung '/', ':', dll.
            mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        elif model_name == "sarima_model":
            import joblib
            joblib.dump(model, "sarima_model.pkl")
            mlflow.log_artifact("sarima_model.pkl", artifact_path="sarima_model")

        print(f"Logged to MLflow: {model_name}")

def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def log_plot(y_true, y_pred, name="plots/forecast_plot.png", title="Prediction vs Actual"):
    _ensure_parent_dir(name)

    plt.figure(figsize=(10, 5))
    x_axis = getattr(y_true, "index", None)
    if isinstance(x_axis, pd.Index):
        plt.plot(x_axis, y_true, label="Actual")
        plt.plot(x_axis, y_pred, label="Prediction")
    else:
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Prediction")

    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

    mlflow.log_artifact(name, artifact_path="plots")
    print(f"Plot saved and logged: {name}")

def plot_forecast(y_true, y_pred, filename="outputs/xgboost_forecast_vs_actual.png", log_to_mlflow=True):
    _ensure_parent_dir(filename)

    plt.figure(figsize=(12, 6))
    x_axis = getattr(y_true, "index", None)
    if isinstance(x_axis, pd.Index):
        plt.plot(x_axis, y_true, label="Actual", linewidth=2)
        plt.plot(x_axis, y_pred, label="Forecast", linestyle="--")
    else:
        plt.plot(y_true.values, label="Actual", linewidth=2)
        plt.plot(y_pred, label="Forecast", linestyle="--")

    plt.title("XGBoost Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Plot disimpan di {filename}")

    if log_to_mlflow:
        mlflow.log_artifact(filename, artifact_path="plots")
        print("Plot juga dicatat ke MLflow.")
