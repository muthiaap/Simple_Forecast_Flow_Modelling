import os
from datetime import datetime

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import fetch_stock_data
from preprocessing import preprocess_stock_data
from model_trainer import train_models_consistent, evaluate_model
from mlops_logger import setup_mlflow, log_metrics_params_model, log_plot

st.set_page_config(layout="centered", page_title="Stock Forecast Demo")
st.title("📈 Simple Forecast for Stock Price")
st.markdown("""
Two models implemented:
- **SARIMA** (statistical time-series)
- **XGBoost** (gradient boosting ML)

Enter stock code (e.g., `BBCA.JK`, `ASII.JK`, `TLKM.JK`) and click predict.
""")

# Sidebar controls
st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test set proportion", 0.05, 0.40, 0.20, 0.01)
use_mlflow = st.sidebar.checkbox("Log to MLflow", value=True)

ticker = st.text_input("Stock code:", value="BBCA.JK")

@st.cache_data(show_spinner=False)
def _fetch(t):
    return fetch_stock_data(t, start="2020-01-01", save_csv=False)

def _features_used(df_proc: pd.DataFrame) -> int:
    return (
        df_proc.drop(columns=["Close", "Date"], errors="ignore")
        .select_dtypes(include="number")
        .shape[1]
    )

if st.button("🔍 Run Prediction"):
    try:
        with st.spinner("Fetching & processing data ..."):
            if use_mlflow:
                setup_mlflow(experiment_name="stock_forecasting")

            df = _fetch(ticker)
            if isinstance(df.index, pd.DatetimeIndex) and "Date" not in df.columns:
                df = df.reset_index().rename(columns={"index": "Date", df.index.name or "Date": "Date"})

            cols = set(df.columns)
            if "Close" not in cols:
                if "Adj Close" in cols:
                    df["Close"] = df["Adj Close"] 
                else:
                    st.error(f"Kolom 'Close' tidak ditemukan. Kolom tersedia: {list(df.columns)}")
                    st.stop()

            df_proc = preprocess_stock_data(df)
            needed = {"Date", "Close"}
            missing = [c for c in needed if c not in df_proc.columns]
            if missing:
                st.error(f"Kolom wajib hilang setelah preprocess: {missing}. Kolom saat ini: {list(df_proc.columns)}")
                st.write("Columns df awal:", list(df.columns))
                st.stop()

            # Train both models
            results = train_models_consistent(df_proc, test_size=test_size)

            # Evaluate
            sarima_metrics = evaluate_model(results["y_test"], results["sarima_pred"], model_name="SARIMA")
            xgb_metrics    = evaluate_model(results["y_test"], results["xgb_pred"],    model_name="XGBoost")

        # Plot
        os.makedirs("plots", exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        test_dates    = pd.to_datetime(results["test_dates"])
        actual_values = results["y_test"].values

        ax.plot(test_dates, actual_values, label="Actual", linewidth=2)
        ax.plot(test_dates, results["xgb_pred"], label="XGBoost", linestyle="--")
        ax.plot(test_dates, results["sarima_pred"], label="SARIMA", linestyle=":")
        ax.set_title(f"Prediksi Harga Penutupan (Test {int(test_size*100)}% terakhir) — {ticker}")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (IDR)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_filename = os.path.join("plots", f"xgboost_vs_sarima_{ticker.replace('.','_')}.png")
        fig.savefig(plot_filename)
        st.pyplot(fig)

        # MLflow logging
        if use_mlflow:
            run_suffix = datetime.today().strftime("%Y%m%d_%H%M%S")

            params_xgb = {
                "model": "XGBoost",
                "framework": "xgboost",
                "stock": ticker,
                "start_date": df_proc["Date"].min().strftime("%Y-%m-%d"),
                "end_date":   df_proc["Date"].max().strftime("%Y-%m-%d"),
                "test_size":  test_size,
                "features_used": _features_used(df_proc),
            }
            run_name_xgb = f"XGB_{ticker}_{run_suffix}"
            log_metrics_params_model(
                results["xgb_model"], params_xgb, xgb_metrics,
                model_name="xgboost_model", run_name=run_name_xgb
            )

            params_sarima = {
                "model": "SARIMA",
                "framework": "statsmodels",
                "stock": ticker,
                "start_date": df_proc["Date"].min().strftime("%Y-%m-%d"),
                "end_date":   df_proc["Date"].max().strftime("%Y-%m-%d"),
                "test_size":  test_size,
                "features_used": 1,
            }
            run_name_sarima = f"SARIMA_{ticker}_{run_suffix}"
            log_metrics_params_model(
                results["sarima_model"], params_sarima, sarima_metrics,
                model_name="sarima_model", run_name=run_name_sarima
            )

            log_plot(results["y_test"], results["xgb_pred"], name=plot_filename,
                     title=f"XGBoost vs Actual — {ticker}")

        st.subheader(f"📊 Grafik Prediksi vs Aktual — {ticker}")

        st.subheader("📋 Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total rows:** {len(df_proc)}")
            st.write(f"**Train rows:** {len(df_proc) - len(results['y_test'])}")
            st.write(f"**Test rows:** {len(results['y_test'])}")
        with col2:
            st.write(f"**Data range:** {df_proc['Date'].min().strftime('%Y-%m-%d')} → {df_proc['Date'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Test period:** {test_dates.iloc[0].strftime('%Y-%m-%d')} → {test_dates.iloc[-1].strftime('%Y-%m-%d')}")
            st.write(f"**Test proportion:** {test_size:.2f}")

        st.subheader("📋 Model Evaluation")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**SARIMA**")
            st.json(sarima_metrics)
        with c2:
            st.markdown("**XGBoost**")
            st.json(xgb_metrics)

        st.subheader("🧾 Last 10 Test Rows")
        latest_data = pd.DataFrame({
            "Tanggal": test_dates,
            "Aktual": actual_values,
            "Prediksi XGBoost": results["xgb_pred"],
            "Prediksi SARIMA": results["sarima_pred"],
        })
        st.dataframe(latest_data.tail(10), use_container_width=True)

        # Direction signal (next day vs last actual)
        st.subheader("📈 Next-Day Direction (Up/Down)")
        curr_actual = float(actual_values[-1])
        next_pred   = float(results["xgb_pred"][-1])
        if next_pred > curr_actual:
            st.success(f"Predicted **UP** next day. Pred {next_pred:.2f} > Last actual {curr_actual:.2f}")
        elif next_pred < curr_actual:
            st.error(f"Predicted **DOWN** next day. Pred {next_pred:.2f} < Last actual {curr_actual:.2f}")
        else:
            st.info(f"Predicted **FLAT** next day. Pred {next_pred:.2f} = Last actual {curr_actual:.2f}")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
