import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def _normalize_df(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # If multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x is not None and x != ""]).strip()
            for tup in df.columns.to_flat_index()
        ]
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    return df

# Split data
def time_series_split(df, test_size=0.2):
    n_total = len(df)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    return train_df, test_df

# Model SARIMA
def train_sarima(df, test_size=0.2, target_col="Close", date_col="Date"):
    df = _normalize_df(df, date_col=date_col)
    train_df, test_df = time_series_split(df, test_size=test_size)
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    model = SARIMAX(y_train, order=(5, 1, 0))
    result = model.fit(disp=False)
    forecast = result.forecast(steps=len(y_test))
    return result, y_test, forecast

# Model XGBoost
def train_xgboost(df, target_col="Close", test_size=0.2, date_col="Date"):
    df = _normalize_df(df, date_col=date_col)
    train_df, test_df = time_series_split(df, test_size=test_size)

    drop_cols = [c for c in [target_col, date_col] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")
    y_test = test_df[target_col]

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

# training model
def train_models_consistent(df, target_col="Close", test_size=0.2, date_col="Date"):
    df = _normalize_df(df, date_col=date_col)
    train_df, test_df = time_series_split(df, test_size=test_size)

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # SARIMA
    sarima_model = SARIMAX(y_train, order=(5, 1, 0))
    sarima_result = sarima_model.fit(disp=False)
    sarima_forecast = sarima_result.forecast(steps=len(y_test))

    # XGBoost
    drop_cols = [c for c in [target_col, date_col] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")
    X_test = test_df.drop(columns=drop_cols, errors="ignore").select_dtypes("number")

    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    return {
        'sarima_model': sarima_result,
        'xgb_model': xgb_model,
        'y_test': y_test,
        'sarima_pred': sarima_forecast,
        'xgb_pred': xgb_pred,
        'test_dates': test_df[date_col]
    }

# Evaluate Model
def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"📊 Evaluasi {model_name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {"RMSE": rmse, "MAE": mae, "R2": r2}
