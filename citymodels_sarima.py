# citymodels_sarima.py (fixed)
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def evaluate_model(df, forecast, city, horizon_hours=168):
    test_df = df.iloc[-horizon_hours:]
    actual = test_df["y"].values
    predicted = forecast[:horizon_hours]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, np.nan, actual))) * 100
    smape = 100/len(actual) * np.sum(
        2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))
    )

    print(f"\n {city} Evaluation:")
    print(f"   MAE   = {mae:.2f} mm")
    print(f"   RMSE  = {rmse:.2f} mm")
    print(f"   MAPE  = {mape:.2f}%")
    print(f"   SMAPE = {smape:.2f}%")

    return mae, rmse, mape, smape


def train_city_model(city, filename, days_ahead=7):
    df = prepare_city_data(filename)
    threshold = calculate_hybrid_threshold(df)
    print(f"🔹 {city}: Hybrid danger threshold set to {threshold:.2f} mm")

    # SARIMA model
    model = SARIMAX(df["y"], order=(2, 1, 2),
                    seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    sarima_fit = model.fit(disp=False)

    # Forecast next N days hourly
    steps = days_ahead * 24
    forecast = sarima_fit.forecast(steps=steps)

    forecast_df = pd.DataFrame({
        "ds": pd.date_range(df["ds"].iloc[-1] + pd.Timedelta(hours=1), periods=steps, freq="h"),
        "yhat": forecast
    })
    forecast_df["DangerZone"] = forecast_df["yhat"].apply(
        lambda y: "⚠️ Danger" if y > threshold else "✅ Safe"
    )

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"].iloc[-7*24:], df["y"].iloc[-7*24:], label="Actual (last 7d)")
    plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast (next 7d)")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.0f} mm")
    plt.legend()
    plt.title(f"{city} - 7 Day Hourly Forecast (SARIMA)")
    plot_filename = f"plots/{city}_forecast_sarima.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    # Save forecast CSV
    output_csv = f"forecast_{city}_sarima.csv"
    forecast_df.to_csv(output_csv, index=False)

    print(f" Saved forecast plot: {plot_filename}")
    print(f" Saved forecast data: {output_csv}")
    print(forecast_df.tail())  # print once

    # Evaluate on last 7d
    sarima_fit_forecast = sarima_fit.forecast(steps=7*24)
    mae, rmse, mape, smape = evaluate_model(df, sarima_fit_forecast, city)

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
        metrics = train_city_model(city, filename, days_ahead=7)
        results.append(metrics)

    acc_df = pd.DataFrame(results)
    acc_df.to_csv("model_accuracy_sarima.csv", index=False)
    print("\n Saved SARIMA model accuracy scores to model_accuracy_sarima.csv")
    print(acc_df)
