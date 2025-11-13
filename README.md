# Ocean Level Rise Detection  
Sea-Level Forecasting for Coastal Cities using Time-Series Models

## Overview  
This project focuses on forecasting sea level variations in major coastal cities using historical tide gauge data. Multiple time-series models are implemented and compared to evaluate their effectiveness in predicting future sea-level trends.

## Objectives  
- Develop and compare predictive models for sea-level forecasting.  
- Evaluate statistical and machine-learning approaches.  
- Generate forecast outputs for multiple coastal cities.  
- Support coastal planning, disaster management, and environmental monitoring.

## Motivation  
Rising sea levels pose long-term risks to coastal regions, impacting infrastructure, communities, and ecosystems. Accurate forecasts help with:  
- Flood risk prediction  
- Urban and coastal infrastructure planning  
- Environmental impact assessment  
- Early warning systems

## Project Structure  
/
│ README.md
│ citymodels.py
│ citymodels_lstm.py
│ citymodels_sarima.py
│ citymodels_xgb.py
│ dashboard_xgb.py
│ new_realtime_fetch.py
│ historical_<City>.csv
│ forecast_<City><Model>.csv
│ model_accuracy*.csv
│
└── plots/
└── <City>_LSTM_model.pkl

## Data  
- Historical tidal datasets for coastal US cities.  
- Files follow the naming pattern:  
  - `historical_<City>_<State>.csv` (input time series)  
  - `forecast_<City>_<State>_<Model>.csv` (prediction outputs)  
- Model accuracy comparison files:  
  - `model_accuracy_LSTM.csv`  
  - `model_accuracy_sarima.csv`  
  - `model_accuracy_xgb.csv`

## Models Implemented  

### 1. LSTM (Long Short-Term Memory)  
- Implemented in `citymodels_lstm.py`.  
- Learns sequential temporal dependencies.  
- Generates multi-step forecasts.

### 2. SARIMA (Seasonal ARIMA)  
- Implemented in `citymodels_sarima.py`.  
- Captures autoregressive, differencing, moving-average, and seasonal behaviour.  
- Suitable for short-term cyclical patterns.

### 3. XGBoost (Extreme Gradient Boosting)  
- Implemented in `citymodels_xgb.py` and `dashboard_xgb.py`.  
- Uses lag features, rolling window statistics, and non-linear modelling.  
- Demonstrated best performance across most cities.

## Workflow  
1. Load historical sea-level dataset for each city.  
2. Preprocess (cleaning, resampling, feature engineering).  
3. Split into training/testing sets.  
4. Train LSTM, SARIMA, and XGBoost models per city.  
5. Compare RMSE/MAE metrics.  
6. Export forecasts and metrics into CSV files.  
7. Save models and plots into `/plots`.

## Results Summary  
- **XGBoost** achieved the best overall accuracy and generalization.  
- **SARIMA** effectively captured seasonal tidal patterns but struggled with long-range forecasts.  
- **LSTM** performed well with longer sequences but required more tuning and larger datasets.  
- XGBoost chosen as the final primary model for production-level forecasting.

## How to Use  

### 1. Clone the Repository  
```bash
git clone https://github.com/AmulyaKaushik/Ocean-Level-Rise-Detection.git
```
### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn keras
```
### 3. Run Model Scripts
```bash
python citymodels_xgb.py
python citymodels_lstm.py
python citymodels_sarima.py
```
### 4. View Output Files
Forecast results appear as:
forecast_<City>_<State>_<Model>.csv

Accuracy files:
model_accuracy_<model>.csv

### 5. Dashboard
```bash
python dashboard_xgb.py
```
### 6. Real-Time Updates
```bash
python new_realtime_fetch.py
```
### Future Enhancements
- Add weather/oceanographic external regressors.
- Hyperparameter optimization for all models.
- Implement GRU, Temporal CNN, or Transformer-based models.
- Build a full interactive forecasting dashboard.
- Integrate global tidal datasets.
- Automate data refresh and model retraining.

### Contributors
- Amulya Kaushik — Model development, analysis, documentation
- Shreshtha1804 — Data preparation, code contribution

### License
This project is part of the Minor Project coursework. Code may be reused for educational purposes with proper credit.