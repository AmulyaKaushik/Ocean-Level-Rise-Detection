import pandas as pd
import numpy as np

FILES = {
    "LSTM": "model_accuracy_LSTM.csv",
    "SARIMA": "model_accuracy_sarima.csv",
    "XGBOOST": "model_accuracy_xgb.csv"
}

CITIES = ["San_Francisco_CA", "New_Orleans_LA"]


def load_thresholds():
    """
    Load thresholds from the LSTM file because it contains the real threshold values.
    """
    lstm_df = pd.read_csv("model_accuracy_LSTM.csv")
    thresholds = dict(zip(lstm_df["City"], lstm_df["Threshold (mm)"]))
    return thresholds


def normalize(df, model_name, thresholds):
    df = df.copy()

    # Case 1: LSTM format is already correct
    if "MAE (mm)" in df.columns:
        df["Model"] = model_name
        return df

    # Case 2: SARIMA / XGBOOST have MAE, RMSE, MAPE, SMAPE
    df.rename(columns={
        "MAE": "MAE (mm)",
        "RMSE": "RMSE (mm)"
    }, inplace=True)

    # Add threshold from LSTM
    df["Threshold (mm)"] = df["City"].map(thresholds)

    # Calculate accuracy using threshold
    df["Accuracy (%)"] = (1 - (df["MAE (mm)"] / df["Threshold (mm)"])) * 100

    # Drop irrelevant columns
    for col in ["MAPE", "SMAPE"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df["Model"] = model_name
    return df


def compare_models():
    thresholds = load_thresholds()
    combined = []

    for model_name, file_path in FILES.items():
        df = pd.read_csv(file_path)

        df = normalize(df, model_name, thresholds)
        df = df[df["City"].isin(CITIES)]

        combined.append(df)

    final_df = pd.concat(combined, ignore_index=True)

    final_df = final_df[[
        "Model",
        "City",
        "MAE (mm)",
        "RMSE (mm)",
        "Accuracy (%)",
        "Threshold (mm)"
    ]]

    # Print city-wise
    for city in CITIES:
        print("\n====================================")
        print(f"          CITY: {city}")
        print("====================================")
        print(final_df[final_df["City"] == city][[
            "Model", "MAE (mm)", "RMSE (mm)", "Accuracy (%)"
        ]])
        print()

    final_df.to_csv("comparison_two_cities.csv", index=False)
    print("\nSaved: comparison_two_cities.csv")

    return final_df


if __name__ == "__main__":
    compare_models()
