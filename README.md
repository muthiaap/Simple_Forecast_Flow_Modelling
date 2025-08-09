# Simple Stock Price Forecasting with SARIMA & XGBoost + MLflow

This project predicts stock closing prices using two models:
- **SARIMA** → Statistical time series model (seasonal ARIMA).
- **XGBoost** → Machine learning gradient boosting model.
All runs, metrics, plots, and models are tracked in MLflow.

## 🚀 Features
- **Automatic Data Fetching** from Yahoo Finance using `yfinance`.
- **Preprocessing**:
  - Lag features (`Close_t-1`, `Close_t-2`, etc.)
  - Moving Average & Rolling Std
  - Relative Strength Index (RSI)
- **Two Models**: SARIMA & XGBoost.
- **Model Evaluation** with RMSE, MAE, and R².
- **MLflow Logging**: parameters, metrics, plots, and environment.
- **Interactive Web UI** using Streamlit.

## 🔧 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/username/repo.git
   cd repo

2. Create virtual environment & install dependencies
   ```bash
   python -m venv fm_env
   source fm_env/bin/activate   # macOS/Linux
   fm_env\Scripts\activate      # Windows
   
   pip install -r requirements.txt
   Start MLflow UI
   mlflow ui
3. Open in browser: http://127.0.0.1:5000

## How to Run
1. CLI Mode (main.py)
   ```bash
   python main.py
   Default ticker: BBCA.JK

2. Web UI (Streamlit)
   ```bash
   streamlit run ui_app.py

## Output
Actual vs Predicted price chart (SARIMA & XGBoost).
Evaluation metrics (RMSE, MAE, R²).
Next day price movement prediction (up, down, same).
All artifacts saved in MLflow.

## Notes
Each run trains a new model because stock characteristics differ.
For sector-level models, additional grouping and testing is required.