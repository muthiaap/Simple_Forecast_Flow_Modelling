import os
import random
import subprocess
from datetime import datetime
import argparse
import numpy as np

from data_loader import fetch_stock_data
from preprocessing import preprocess_stock_data
from model_trainer import train_xgboost, evaluate_model
from mlops_logger import setup_mlflow, log_metrics_params_model, log_plot
import mlflow

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_and_log_environment(outfile="requirements_run.txt"):
    reqs = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    with open(outfile, "w") as f:
        f.write(reqs)
    mlflow.log_artifact(outfile)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="BBCA.JK", help="Kode saham, ex: BBCA.JK")
    p.add_argument("--start_date", default="2020-01-01", help="Tanggal mulai data")
    p.add_argument("--test_size", type=float, default=0.2, help="Porsi test (0-1)")
    p.add_argument("--n_estimators", type=int, default=100, help="Jumlah trees XGBoost")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()
    set_random_seed(args.seed)
    os.makedirs("plots", exist_ok=True)

    # 1) Setup MLflow
    setup_mlflow(experiment_name="stock_forecasting")

    # 2) Get data
    ticker = args.ticker
    print(f"📥 Mengambil data saham {ticker}...")
    df_raw = fetch_stock_data(ticker, start=args.start_date, save_csv=False)

    # 3) Preprocessing & feature engineering
    print("🧪 Preprocessing data...")
    df_proc = preprocess_stock_data(df_raw)

    # 4) Train model XGBoost 
    print("🤖 Melatih model XGBoost...")
    model, X_test, y_test, y_pred = train_xgboost(
        df_proc,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )

    # 5) Evaluate
    print("📊 Mengevaluasi model...")
    metrics = evaluate_model(y_test, y_pred, model_name="XGBoost")

    # 6) Logging to MLflow
    print("📝 Logging to MLflow...")
    params = {
        "model": "XGBoost",
        "framework": "xgboost",
        "n_estimators": args.n_estimators,
        "features_used": int(len(X_test.columns)),
        "stock": ticker,
        "start_date": args.start_date,
        "test_size": args.test_size,
        "random_seed": args.seed,
    }
    run_name = f"XGBOOST_{ticker.replace('.', '_')}_{datetime.today().strftime('%Y%m%d')}"
    log_metrics_params_model(
        model, params, metrics, model_name="xgboost_model", run_name=run_name
    )

    plot_path = f"plots/xgboost_forecast_{ticker.replace('.', '_')}.png"
    log_plot(y_test, y_pred, name=plot_path)

    save_and_log_environment()

    print("✅ Semua selesai!")


if __name__ == "__main__":
    main()
