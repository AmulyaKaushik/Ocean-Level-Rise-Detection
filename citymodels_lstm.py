import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import os


def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


def train_lstm_model(city, city_csv_path, epochs=40, batch_size=32, days_ahead=7, eval_days=30):
    # City-specific time steps for tuning
    if "Francisco" in city:
        time_steps = 48
    else:
        time_steps = 30

    # Load dataset
    data = pd.read_csv(city_csv_path)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric column found in dataset.")
    target_col = numeric_cols[0]

    # Convert from meters → millimeters
    values = data[target_col].values.reshape(-1, 1) * 1000

    # Scale between (-1, 1) to preserve oscillation amplitude
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)

    # Create training sequences
    X, y = create_sequences(scaled, time_steps)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build model
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.15),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.1, verbose=1,
        callbacks=[early_stop, reduce_lr]
    )

    # Evaluate
    pred = model.predict(X_test)
    pred_rescaled = scaler.inverse_transform(pred)
    y_test_rescaled = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, pred_rescaled))
    mean_actual = np.mean(y_test_rescaled)
    accuracy = 100 * (1 - (mae / mean_actual))

    print(f"\n📊 Evaluation Results for {city}")
    print(f"MAE: {mae:.2f} mm")
    print(f"RMSE: {rmse:.2f} mm")
    print(f"Accuracy: {accuracy:.2f}%")

    # Threshold
    threshold = max(np.quantile(values, 0.95), np.mean(values) + 2 * np.std(values))

    # Forecast (hybrid stabilized loop)
    future_steps = days_ahead * 24
    last_sequence = scaled[-time_steps:].copy()
    future_predictions = []
    alpha = 0.85  # higher to preserve amplitude

    for step in range(future_steps):
        pred = model.predict(last_sequence.reshape(1, time_steps, 1), verbose=0)
        pred_val = np.clip(pred[0, 0], -1, 1)

        # smooth but maintain oscillation
        pred_val = (alpha * pred_val + (1 - alpha) * float(last_sequence[-1])).item()

        # add tiny random noise to avoid flat predictions
        pred_val += np.random.normal(0, 0.01)

        future_predictions.append(pred_val)

        # hybrid input update — use actuals initially, then predictions
        if step < len(y_test):
            last_sequence = np.append(last_sequence[1:], scaled[-len(y_test) + step][0])
        else:
            last_sequence = np.append(last_sequence[1:], pred_val)

    # Inverse transform
    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Generate future timestamps
    last_date = pd.to_datetime(data['date'].iloc[-1]) if 'date' in data.columns else pd.Timestamp.today()
    future_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(future_steps)]

    # Save forecast CSVs
    eval_forecast_df = pd.DataFrame({
        'Actual (mm)': y_test_rescaled.flatten(),
        'Predicted (mm)': pred_rescaled.flatten()
    })
    eval_forecast_df.to_csv(f"forecast_{city}_LSTM.csv", index=False)

    future_forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Value (mm)': future_predictions_rescaled.flatten()
    })
    future_forecast_df.to_csv(f"forecast_future_{city}_LSTM.csv", index=False)

    # Save model
    with open(f"{city}_LSTM_model.pkl", 'wb') as f:
        pickle.dump(model, f)

    # Plot
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(-eval_days * 24, 0), y_test_rescaled[-eval_days * 24:], label=f"Actual (last {eval_days}d)", color="blue")
    plt.plot(np.arange(0, future_steps), future_predictions_rescaled, label=f"Forecast (next {days_ahead}d)", color="orange")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.0f} mm")
    plt.legend()

    plt.title(f"{city} - Evaluation ({eval_days}d) + Forecast ({days_ahead}d)\n"
              f"MAE={mae:.2f} | RMSE={rmse:.2f} | Accuracy={accuracy:.2f}%")
    plt.ylabel("Value (mm)")
    plt.xlabel("Time (hours)")
    plt.savefig(f"plots/{city}_forecast_LSTM.png", dpi=300)
    plt.close()

    print(f"\n✅ Saved forecast plot: plots/{city}_forecast_LSTM.png")
    print(f"💾 Saved CSVs and model for {city}\n")

    return mae, rmse, accuracy, threshold


if __name__ == "__main__":
    results = []
    for city, city_file in [
        ("San_Francisco_CA", "historical_San_Francisco_CA.csv"),
        ("New_Orleans_LA", "historical_New_Orleans_LA.csv")
    ]:
        if os.path.exists(city_file):
            mae, rmse, accuracy, threshold = train_lstm_model(city, city_file)
            results.append({
                "City": city,
                "MAE (mm)": round(mae, 2),
                "RMSE (mm)": round(rmse, 2),
                "Accuracy (%)": round(accuracy, 2),
                "Threshold (mm)": round(threshold, 2)
            })
        else:
            print(f"⚠️ File {city_file} not found.")

    if results:
        acc_df = pd.DataFrame(results)
        acc_df.to_csv("model_accuracy_LSTM.csv", index=False)
        print("\n💾 Saved LSTM model accuracy scores to model_accuracy_LSTM.csv")
        print(acc_df)
