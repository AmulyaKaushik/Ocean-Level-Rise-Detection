# city_models.py
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import numpy as np


def prepare_city_data(filename):
    """
    Load historical NOAA data from CSV and format for Prophet.
    CSV must have columns: 't' (timestamp), 'v' (water level in meters).
    Converts to mm and renames for Prophet (ds, y).
    """
    df = pd.read_csv(filename)
    df["t"] = pd.to_datetime(df["t"])
    df = df[["t", "v"]].dropna()
    df = df.rename(columns={"t": "ds", "v": "y"}) 
    # Convert meters → millimeters
    df["y"] = df["y"] * 1000
    return df


def calculate_hybrid_threshold(df):
    """
    Hybrid danger threshold = max(95th percentile, mean + 2*std).
    """
    perc95 = df["y"].quantile(0.95)
    std_based = df["y"].mean() + 2 * df["y"].std()
    return max(perc95, std_based)


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(df, city):
    """
    Evaluate Prophet forecast accuracy on last 65 days of data.
    Reports MAE, RMSE, MAPE, SMAPE.
    Accuracy (%) = max(0, 100 - MAPE).
    """
    # Train/test split
    train_df = df.iloc[:-65]
    test_df = df.iloc[-65:]

    # Prophet model
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.5
    )
    model.add_seasonality(name="weekly", period=7, fourier_order=5)
    model.fit(train_df)

    # Forecast over test period
    future = model.make_future_dataframe(periods=65, freq="D")
    forecast = model.predict(future)

    # Align predictions with actual test data
    pred = forecast[["ds", "yhat"]].set_index("ds").loc[test_df["ds"]]
    actual = test_df.set_index("ds")["y"]

    # --- Metrics ---
    mae = mean_absolute_error(actual, pred["yhat"])
    rmse = np.sqrt(mean_squared_error(actual, pred["yhat"]))
    # Safe MAPE (ignores zero denominators)
    mape = np.mean(np.abs((actual - pred["yhat"]) / actual.replace(0, np.nan))) * 100
    # Symmetric MAPE
    smape = 100/len(actual) * np.sum(
        2 * np.abs(pred["yhat"] - actual) / (np.abs(actual) + np.abs(pred["yhat"]))
    )

    # Final accuracy (clamped so never negative)
    accuracy = max(0, 100 - mape)

    print(f"\n {city} Evaluation:")
    print(f"   MAE   = {mae:.2f} mm")
    print(f"   RMSE  = {rmse:.2f} mm")
    print(f"   MAPE  = {mape:.2f}%")
    print(f"   SMAPE = {smape:.2f}%")
    print(f"   Accuracy ≈ {accuracy:.2f}%")

    return accuracy


def train_city_model(city, filename, days_ahead=7):
    """
    Train Prophet model on city historical (365-day hourly) data and forecast next N days.
    Flags 'Danger Zone' if predicted sea level > hybrid threshold.
    Saves forecast plot + forecast CSV.
    """
    df = prepare_city_data(filename)

    # Resample hourly → daily mean for Prophet
    df = df.set_index("ds").resample("1D").mean().reset_index()

    # Calculate hybrid threshold
    threshold = calculate_hybrid_threshold(df)
    print(f"🔹 {city}: Hybrid danger threshold set to {threshold:.2f} mm")

    # Prophet with tuned parameters
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.5
    )
    model.add_seasonality(name="weekly", period=7, fourier_order=5)

    model.fit(df)

    # Forecast next N days
    future = model.make_future_dataframe(periods=days_ahead, freq="D")
    forecast = model.predict(future)

    # Add danger classification
    forecast["DangerZone"] = forecast["yhat"].apply(
        lambda y: "⚠️ Danger" if y > threshold else "✅ Safe"
    )

    # Plot forecast
    os.makedirs("plots", exist_ok=True)
    fig = model.plot(forecast)
    plt.title(f"{city} - {days_ahead} Day Forecast (Hybrid Threshold={threshold:.0f} mm)")
    plot_filename = f"plots/{city}_forecast.png"
    fig.savefig(plot_filename, dpi=300)

    # Save forecast with classification
    output_csv = f"forecast_{city}.csv"
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "DangerZone"]].tail(days_ahead).to_csv(output_csv, index=False)

    print(f" Saved forecast plot: {plot_filename}")
    print(f" Saved forecast data with danger classification: {output_csv}")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "DangerZone"]].tail(days_ahead))

    return df, forecast


if __name__ == "__main__":
    results = []

    for city, filename in [
        ("San_Francisco_CA", "historical_San_Francisco_CA.csv"),
        ("New_Orleans_LA", "historical_New_Orleans_LA.csv")
    ]:
        # Train & forecast
        df, forecast = train_city_model(city, filename, days_ahead=7)

        # Evaluate accuracy
        accuracy = evaluate_model(df, city)
        results.append({"City": city, "Accuracy (%)": round(accuracy, 2)})

    # Save accuracy results
    acc_df = pd.DataFrame(results)
    acc_df.to_csv("model_accuracy.csv", index=False)
    print("\nSaved model accuracy scores to model_accuracy.csv")
    print(acc_df)
