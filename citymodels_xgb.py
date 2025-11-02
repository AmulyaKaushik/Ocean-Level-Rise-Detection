# citymodels_xgb.py (Hybrid, Clean Forecast)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_city_data(filename):
    df = pd.read_csv(filename)
    df["t"] = pd.to_datetime(df["t"])
    df = df[["t", "v"]].dropna()
    df = df.rename(columns={"t": "ds", "v": "y"})
    df["y"] = df["y"] * 1000  # meters → mm
    df = df.set_index("ds").resample("1h").mean().interpolate().reset_index()
    return df


def calculate_hybrid_threshold(df):
    perc95 = df["y"].quantile(0.95)
    std_based = df["y"].mean() + 2 * df["y"].std()
    return max(perc95, std_based)


def create_features(df, lags, rolling_windows):
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for window in rolling_windows:
        df[f"roll_mean_{window}"] = df["y"].rolling(window).mean()
        df[f"roll_std_{window}"] = df["y"].rolling(window).std()

    df = df.dropna().reset_index(drop=True)
    return df


def evaluate_forecast(actual, predicted, city):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, np.nan, actual))) * 100
    smape = 100/len(actual) * np.sum(
        2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))
    )

    print(f"\n {city} Evaluation (last 30 days):")
    print(f"   MAE   = {mae:.2f} mm")
    print(f"   RMSE  = {rmse:.2f} mm")
    print(f"   MAPE  = {mape:.2f}%")
    print(f"   SMAPE = {smape:.2f}%")

    return mae, rmse, mape, smape


def train_city_model(city, filename, days_ahead=7, eval_days=30):
    # City-specific configs
    if city == "San_Francisco_CA":
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
        rolling_windows = [24, 72, 168]
        model = XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        history_window = 168

    else:  # New Orleans
        lags = [1, 2, 3, 6, 12, 24, 48, 72]
        rolling_windows = [24, 72]
        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.7, colsample_bytree=0.7,
            reg_lambda=1.0, reg_alpha=0.5, random_state=42
        )
        history_window = 168

    df = prepare_city_data(filename)
    threshold = calculate_hybrid_threshold(df)
    print(f"🔹 {city}: Hybrid danger threshold set to {threshold:.2f} mm")

    # Features
    df = create_features(df, lags, rolling_windows)

    # Train/test split for evaluation
    eval_horizon = eval_days * 24
    train_df = df.iloc[:-eval_horizon]
    test_df = df.iloc[-eval_horizon:]

    X_train, y_train = train_df.drop(columns=["ds", "y"]), train_df["y"]
    X_test, y_test = test_df.drop(columns=["ds", "y"]), test_df["y"]

    # Train
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae, rmse, mape, smape = evaluate_forecast(y_test, preds, city)

    # Recursive forecast
    forecast_horizon = days_ahead * 24
    last_known = df.iloc[-history_window:].copy()
    forecast = []

    for i in range(forecast_horizon):
        feature_row = {}
        for lag in lags:
            feature_row[f"lag_{lag}"] = last_known["y"].iloc[-lag]
        for window in rolling_windows:
            feature_row[f"roll_mean_{window}"] = last_known["y"].iloc[-window:].mean()
            feature_row[f"roll_std_{window}"] = last_known["y"].iloc[-window:].std()

        X_future = pd.DataFrame([feature_row])
        pred = model.predict(X_future)[0]

        future_time = last_known["ds"].iloc[-1] + pd.Timedelta(hours=1)
        forecast.append((future_time, pred))

        last_known = pd.concat(
            [last_known, pd.DataFrame({"ds": [future_time], "y": [pred]})],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(forecast, columns=["ds", "yhat"])
    forecast_df["DangerZone"] = forecast_df["yhat"].apply(
        lambda y: "⚠️ Danger" if y > threshold else "✅ Safe"
    )

    # Plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"].iloc[-eval_horizon:], df["y"].iloc[-eval_horizon:], label=f"Actual (last {eval_days}d)")
    # plt.plot(test_df["ds"], preds, label="Predicted (last 30d)", color="green", alpha=0.7)
    plt.plot(forecast_df["ds"], forecast_df["yhat"], label=f"Forecast (next {days_ahead}d)", color="orange")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.0f} mm")
    plt.legend()
    plt.title(f"{city} - Evaluation ({eval_days}d) + Forecast ({days_ahead}d)")
    plot_filename = f"plots/{city}_forecast_xgb.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    # Save forecast
    output_csv = f"forecast_{city}_xgb.csv"
    forecast_df.to_csv(output_csv, index=False)

    print(f" Saved forecast plot: {plot_filename}")
    print(f" Saved forecast data: {output_csv}")

    return {
        "City": city,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "SMAPE": round(smape, 2)
    }


if __name__ == "__main__":
    results = []
    for city, filename in [
        ("San_Francisco_CA", "historical_San_Francisco_CA.csv"),
        ("New_Orleans_LA", "historical_New_Orleans_LA.csv")
    ]:
        metrics = train_city_model(city, filename, days_ahead=7, eval_days=30)
        results.append(metrics)

    acc_df = pd.DataFrame(results)
    acc_df.to_csv("model_accuracy_xgb.csv", index=False)
    print("\n💾 Saved XGBoost model accuracy scores to model_accuracy_xgb.csv")
    print(acc_df)
