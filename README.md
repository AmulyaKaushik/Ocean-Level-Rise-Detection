# Sea Level Prediction using Time Series Forecasting Models 🌊

## 📘 Overview
This project focuses on **forecasting sea level changes** for coastal cities using time series modeling techniques.  
We used NOAA (National Oceanic and Atmospheric Administration) sea level data to train and evaluate multiple forecasting models — **Prophet**, **SARIMA**, and **XGBoost** — to understand their strengths and weaknesses in predicting future sea level trends.

---

## 🎯 Objective
To develop and compare different predictive models for sea level forecasting, and to identify the most accurate and robust approach for short-term and long-term water level prediction.

---

## 🧠 Project Motivation
Rising sea levels due to climate change pose significant risks to coastal cities.  
Accurate sea-level prediction helps in:
- Coastal infrastructure planning  
- Flood risk management  
- Environmental monitoring  

This project aims to model the sea level behavior using historical data and forecast future trends using data-driven techniques.

---


## 🧩 Models Implemented

### 1. **Prophet Model (Facebook Prophet)**
**Concept:** Decomposes time series into trend, seasonality, and noise components.

**Parameters Used:**
- `daily_seasonality=True`
- `weekly_seasonality=True`
- `yearly_seasonality=True`
- `changepoint_prior_scale=0.5`
- Added custom weekly seasonality with `fourier_order=5`

**Pros:**
- Automatically detects trend and seasonality.
- Easy to use and interpret.

**Cons:**
- Very sensitive to missing data and noise.
- Poor long-term stability for sea-level data.
- Failed to handle irregular tidal cycles effectively.

---

### 2. **SARIMA Model (Seasonal ARIMA)**
**Concept:** Statistical model combining autoregressive (AR), differencing (I), and moving average (MA) with seasonal components.

**Parameters Used:**
- `order=(2,1,2)`
- `seasonal_order=(1,1,1,24)` (captures 24-hour tidal seasonality)
- `enforce_stationarity=False`
- `enforce_invertibility=False`

**Pros:**
- Captures daily cyclic patterns effectively.
- Stable for short-term forecasting.

**Cons:**
- Limited flexibility — assumes linear relationships.
- Can’t include external regressors (like temperature or pressure).
- Struggles with irregular or long-term patterns.

---

### 3. **XGBoost Model (Extreme Gradient Boosting)**
**Concept:** Machine learning model that builds an ensemble of decision trees sequentially, each learning from the previous one’s errors.

**Features Used:**
- Lag features (previous time steps)
- Rolling statistics (mean, standard deviation)

**Parameters Used:**
- **San Francisco:**
  - `n_estimators=800`, `learning_rate=0.03`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`
- **New Orleans:**
  - `n_estimators=400`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.7`, `colsample_bytree=0.7`, `reg_lambda=1.0`, `reg_alpha=0.5`

**Pros:**
- Handles non-linear dependencies.
- Robust against noisy and irregular data.
- Regularization prevents overfitting.

**Cons:**
- Requires feature engineering (lags, rolling features).
- Hyperparameter tuning needed for best results.

---

## ⚙️ Data Source
- **Dataset:** NOAA (National Oceanic and Atmospheric Administration) Tidal Station Data  
- **Cities Used:** San Francisco, New Orleans  
- **Frequency:** Hourly data, resampled to daily means for Prophet.

---

## 📈 Model Comparison

| **Model** | **Approach** | **Strengths** | **Weaknesses** | **Performance** |
|------------|---------------|----------------|-----------------|-----------------|
| **Prophet** | Trend + Seasonality + Noise | Easy to use, interpretable | Poor with irregular data, unstable long-term | Lowest accuracy |
| **SARIMA** | ARIMA + Seasonal terms | Captures daily cycles, stable | Rigid, no external factors | Better than Prophet |
| **XGBoost** | Gradient Boosted Trees | Non-linear learning, robust, regularized | Requires feature engineering | Best performance |

---

## 🧾 Results Summary
- **Prophet**: Failed to model complex, irregular sea-level fluctuations.  
- **SARIMA**: Captured daily patterns but underperformed in long-term predictions.  
- **XGBoost**: Achieved the lowest RMSE and best overall performance even without weather features.

**→ Final chosen model:** **XGBoost**  
because of its ability to learn complex non-linear patterns and outperform statistical baselines.

---

## 🚀 Future Improvements
- Integrate **weather parameters** (temperature, pressure, humidity, wind speed).  
- Apply **hyperparameter optimization** for XGBoost.  
- Try **LSTM/GRU models** for sequential pattern learning.  
- Deploy as a web-based dashboard for live prediction visualization.

---

## 🧰 Tech Stack
- **Python**  
- **Libraries:** pandas, numpy, xgboost, prophet, statsmodels, matplotlib, scikit-learn  

---

## 📄 License
This project is developed for academic purposes as part of the **Minor Project (Mid-Term)** under [Your College Name].

---

## 👨‍💻 Contributors
- **Amulya Kaushik** — Model development, analysis, and documentation.

---

