# Ocean Level Rise Detection  
Sea-Level Forecasting for Coastal Cities using Time-Series Models

## Overview  
This project focuses on forecasting sea level variations in major coastal cities using historical tide gauge data. Multiple time-series models are implemented and compared to evaluate their effectiveness in predicting future sea-level trends. The repository now includes multi-model comparison tools and an enhanced Streamlit dashboard with tide-station mapping and danger-zone alerts.

## Objectives  
- Develop and compare predictive models for sea-level forecasting.  
- Evaluate statistical and machine-learning approaches (LSTM, SARIMA, XGBoost).  
- Generate forecast outputs for multiple coastal cities.  
- Provide an interactive dashboard with map visualization and alerts for planners and responders.

## Motivation  
Rising sea levels pose long-term risks to coastal regions, impacting infrastructure, communities, and ecosystems. Accurate forecasts help with:  
- Flood risk prediction  
- Urban and coastal infrastructure planning  
- Environmental impact assessment  
- Early warning systems

## Project Structure  
| README.md  
| citymodels.py  
| citymodels_lstm.py  
| citymodels_sarima.py  
| citymodels_xgb.py  
| dashboard.py                (Streamlit dashboard — LSTM view + multi-model comparison & map)  
| dashboard_xgb.py            (XGBoost-specific dashboard helpers)  
| new_realtime_fetch.py  
| COMPARE_ALL_MODELS.PY      (utility to compare LSTM / SARIMA / XGBOOST model accuracy)  
| historical_<City>_<State>.csv  
| forecast_<City>_<State>_<Model>.csv  
| forecast_future_<City>_<State>_<Model>.csv  
| model_accuracy_LSTM.csv  
| model_accuracy_sarima.csv  
| model_accuracy_xgb.csv  

└── plots/  
└── <City>_LSTM_model.pkl

## Data  
- Historical tidal datasets for coastal US cities.  
- Files follow the naming pattern:  
  - `historical_<City>_<State>.csv` (input time series)  
  - `forecast_<City>_<State>_<Model>.csv` (historical evaluation outputs containing Actual/Predicted)  
  - `forecast_future_<City>_<State>_<Model>.csv` (future horizon forecasts with Date and Forecasted columns)  
- Model accuracy comparison files:  
  - `model_accuracy_LSTM.csv`  
  - `model_accuracy_sarima.csv`  
  - `model_accuracy_xgb.csv`

## Models Implemented  

### 1. LSTM (Long Short-Term Memory)  
- Implemented in `citymodels_lstm.py`.  
- Learns sequential temporal dependencies.  
- Generates multi-step forecasts and produces `forecast_<City>_LSTM.csv` and `forecast_future_<City>_LSTM.csv`.  
- Produces `model_accuracy_LSTM.csv` used as the canonical source for threshold values.

### 2. SARIMA (Seasonal ARIMA)  
- Implemented in `citymodels_sarima.py`.  
- Captures autoregressive, differencing, moving-average, and seasonal behaviour.  
- Generates per-city accuracy summaries saved to `model_accuracy_sarima.csv`.

### 3. XGBoost (Extreme Gradient Boosting)  
- Implemented in `citymodels_xgb.py`.  
- Uses lag features, rolling window statistics, and non-linear modelling.  
- Demonstrated best performance across many cities and exports `model_accuracy_xgb.csv`.  
- Dashboard helpers in `dashboard_xgb.py` assist visualizing XGBoost outputs.

## Dashboard (Streamlit)
- `dashboard.py` is the main interactive app. Key features (updated to match current implementation):
  - App layout: multi-page navigation with three pages — "Home", "LSTM Forecast", and "Model Comparison".
  - Dark ocean theme applied via inline CSS for a consistent dark UI.
  - Local Lottie animation loader using the file "Wave Loop - Loading with LottieV2.json" on the Home page.
  - Home page:
    - Shows the Lottie animation, a short overview, and quick stats: number of cities covered, number of cities with available forecasts, and whether multi-model comparison files are present.
  - LSTM Forecast page:
    - Select a city to view historical evaluation (Actual vs Predicted) and future forecasts.
    - Folium-based tide-station map using the "CartoDB dark_matter" tiles and CircleMarkers for NOAA coordinates.
    - Displays the LSTM accuracy table (from model_accuracy_LSTM.csv).
    - Danger-zone alert: compares the maximum future forecast value against the per-city Threshold (mm) from model_accuracy_LSTM.csv and shows an error/success banner.
  - Model Comparison page:
    - Loads LSTM, SARIMA, and XGBoost accuracy CSVs, normalizes fields (MAE, RMSE), and computes Accuracy (%) relative to LSTM thresholds.
    - Comparison UI currently focused on two cities: San_Francisco_CA and New_Orleans_LA, with bar charts for Accuracy and grouped bars for MAE/RMSE.
  - Notes:
    - The Lottie animation file must be present in the project root with the exact filename above.
    - Ensure `model_accuracy_LSTM.csv` is available — it provides the Threshold (mm) values used by the danger-zone alert and accuracy normalization.

- To run the dashboard:
  - pip install streamlit plotly folium streamlit-folium streamlit-lottie
  - streamlit run dashboard.py

## Utilities
- COMPARE_ALL_MODELS.PY — script to normalize and compare model accuracy CSVs and export a compact comparison CSV (example: `comparison_two_cities.csv`). It:
  - Loads thresholds from `model_accuracy_LSTM.csv`.
  - Normalizes SARIMA and XGBOOST formats to a common schema.
  - Computes an "Accuracy (%)" relative to the LSTM-derived threshold.
  - Saves consolidated comparisons for selected cities.

## Workflow  
1. Load historical sea-level dataset for each city.  
2. Preprocess (cleaning, resampling, feature engineering).  
3. Split into training/testing sets.  
4. Train LSTM, SARIMA, and XGBoost models per city.  
5. Export per-model accuracy CSVs (LSTM, SARIMA, XGBOOST).  
6. Use `COMPARE_ALL_MODELS.PY` to aggregate/compare model performances across cities or run interactive comparisons in `dashboard.py`.  
7. Export forecasts and metrics into CSV files.  
8. Save models and plots into `/plots`.

## How to Use  

1. Clone the Repository  
```bash
git clone https://github.com/AmulyaKaushik/Ocean-Level-Rise-Detection.git
```

2. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn keras streamlit plotly folium streamlit-folium
```

3. Real-Time Updates
```bash
python new_realtime_fetch.py
```

4. Run Model Scripts
```bash
python citymodels_xgb.py
python citymodels_lstm.py
python citymodels_sarima.py
```

5. Run comparison utility (example)
```bash
python COMPARE_ALL_MODELS.PY
```

6. Launch Dashboard
```bash
streamlit run dashboard.py
```

### Output files
- Forecast results: `forecast_<City>_<State>_<Model>.csv` and `forecast_future_<City>_<State>_<Model>.csv`  
- Accuracy files: `model_accuracy_LSTM.csv`, `model_accuracy_sarima.csv`, `model_accuracy_xgb.csv`  
- Aggregated comparison: `comparison_two_cities.csv` (or other outputs from the compare script)

## Results Summary  
- **XGBoost** achieved the best overall accuracy and generalization in many cities.  
- **SARIMA** effectively captured seasonal tidal patterns but struggled with long-range forecasts.  
- **LSTM** performed well with longer sequences but required more tuning and larger datasets.  
- Multi-model comparison tools help choose models per city and visualize risk vs threshold.

## Future Enhancements
- Add weather/oceanographic external regressors.  
- Hyperparameter optimization for all models.  
- Implement GRU, Temporal CNN, or Transformer-based models.  
- Build a full interactive forecasting dashboard with scenario simulations.  
- Integrate global tidal datasets.  
- Automate data refresh and model retraining.

## Contributors
- Amulya Kaushik — Model development, analysis, documentation  
- Shreshtha1804 — Data preparation, code contribution

## License
This project is part of the Minor Project coursework. Code may be reused for educational purposes with proper credit.