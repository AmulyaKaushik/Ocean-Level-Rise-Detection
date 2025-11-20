import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os
import requests
from streamlit_lottie import st_lottie
import json

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Ocean Level Rise – Forecast Dashboard", layout="wide")

# ------------------------------------------------------
# DARK OCEAN THEME + FADE-IN ANIMATIONS
# ------------------------------------------------------
CSS = """
<style>
body { background-color:#041427; color:#e6f0f6; }

/* Fade-in container */
.fade-in { opacity:0; animation: fadeIn 0.8s ease forwards; }
@keyframes fadeIn { 0% {opacity:0; transform:translateY(12px);} 100% {opacity:1; transform:translateY(0);} }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#021622,#04243b);
}

/* Titles */
.big-title { font-size:36px; font-weight:800; color:#e6f0f6; }
.subheader { font-size:22px; font-weight:700; color:#bfe6ff; margin-top:18px; }

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#2b9fe6,#2a6bff);
    color:white;
    border-radius:8px;
    border:none;
}

/* DataFrame Header */
.stDataFrame thead th {
    background: rgba(42,107,255,0.18);
    color: #e6f0f6;
}

/* Footer */
.footer { text-align:center; color:#98cfe8; margin-top:20px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------
# LOTTIE LOADER FOR LOCAL FILE
# ------------------------------------------------------
def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        return None

# 🌊 Your local animation file
LOTTIE_PATH = "Wave Loop - Loading with LottieV2.json"

# ------------------------------------------------------
# CITY CONSTANTS
# ------------------------------------------------------
CITIES = [
    "San_Francisco_CA","Los_Angeles_CA","Seattle_WA","Miami_FL",
    "New_York_NY","Boston_MA","New_Orleans_LA","Galveston_TX","Honolulu_HI"
]

CITY_COORDS = {
    "San_Francisco_CA": (37.806, -122.465),
    "Los_Angeles_CA": (33.720, -118.271),
    "Seattle_WA": (47.603, -122.338),
    "Miami_FL": (25.774, -80.13),
    "New_York_NY": (40.700, -74.014),
    "Boston_MA": (42.353, -71.051),
    "New_Orleans_LA": (29.383, -89.433),
    "Galveston_TX": (29.312, -94.797),
    "Honolulu_HI": (21.306, -157.867)
}

# CSV paths
LSTM_ACC = "model_accuracy_LSTM.csv"
SARIMA_ACC = "model_accuracy_sarima.csv"
XGB_ACC = "model_accuracy_xgb.csv"

def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📈 LSTM Forecast", "🔍 Model Comparison"])

st.sidebar.markdown("---")
st.sidebar.caption("Ocean Theme Dashboard 🌊")
st.sidebar.markdown("---")

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if page == "🏠 Home":

    st.markdown("<div class='big-title fade-in'>🌊 Ocean Level Rise Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader fade-in'>Interactive ocean-level forecasts — LSTM & multi-model comparison</div>", unsafe_allow_html=True)

    # Load & display animation
    lottie_anim = load_lottie_file(LOTTIE_PATH)
    if lottie_anim:
        st_lottie(lottie_anim, height=300, key="ocean_anim")
    else:
        st.error("Animation failed to load. Make sure the file name is correct.")

    st.markdown("<div class='subheader'>What this dashboard shows</div>", unsafe_allow_html=True)
    st.write("""
    - LSTM historical / future forecasts per city  
    - Multi-model accuracy comparison (LSTM, SARIMA, XGBoost) for **San Francisco & New Orleans**  
    - Danger-zone alert when forecast levels exceed the threshold  
    """)

    # Stats Row
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cities covered", len(CITIES))
    with col2:
        count_hist = sum(os.path.exists(f"forecast_{c}_LSTM.csv") for c in CITIES)
        st.metric("Cities with forecast", count_hist)
    with col3:
        ready = "Yes" if (os.path.exists(LSTM_ACC) and os.path.exists(SARIMA_ACC) and os.path.exists(XGB_ACC)) else "No"
        st.metric("Comparison ready", ready)

# ------------------------------------------------------
# LSTM FORECAST PAGE
# ------------------------------------------------------
elif page == "📈 LSTM Forecast":

    st.markdown("<div class='big-title fade-in'>📈 LSTM Forecast Dashboard</div>", unsafe_allow_html=True)

    selected_city = st.selectbox("Select City", CITIES)

    hist_file = f"forecast_{selected_city}_LSTM.csv"
    future_file = f"forecast_future_{selected_city}_LSTM.csv"
    acc_df = safe_read(LSTM_ACC)

    st.markdown("<div class='subheader'>📊 LSTM Model Accuracy</div>", unsafe_allow_html=True)
    if acc_df is not None:
        st.dataframe(acc_df)
    else:
        st.error("model_accuracy_LSTM.csv not found!")

    # Map
    st.markdown("<div class='subheader'>📍 Tide Stations Map</div>", unsafe_allow_html=True)
    m = folium.Map(location=[37, -97], zoom_start=4, tiles="CartoDB dark_matter")
    for city, (lat, lon) in CITY_COORDS.items():
        folium.CircleMarker(
            location=[lat, lon], radius=6,
            color="#2a6bff", fill=True, fill_color="#2a9fe6", fill_opacity=0.9,
            popup=city.replace("_"," ")
        ).add_to(m)
    st_folium(m, width=700, height=400)

    # Historical
    st.markdown(f"<div class='subheader'>📉 Historical Evaluation — {selected_city}</div>", unsafe_allow_html=True)
    df_hist = safe_read(hist_file)
    if df_hist is None:
        st.error(f"Missing: {hist_file}")
    else:
        df_hist["Step"] = range(len(df_hist))
        fig_hist = px.line(
            df_hist, x="Step", y=["Actual (mm)", "Predicted (mm)"],
            title=f"{selected_city} – Historical Forecast",
            color_discrete_sequence=["#9ad3ff", "#2a6bff"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Future
    st.markdown(f"<div class='subheader'>🔮 Future Forecast — {selected_city}</div>", unsafe_allow_html=True)
    df_future = safe_read(future_file)
    if df_future is None:
        st.info("No future forecast available.")
    else:
        col = [c for c in df_future.columns if "Forecasted" in c][0]
        fig_future = px.line(
            df_future, x="Date", y=col,
            title=f"{selected_city} – Future Forecast",
            color_discrete_sequence=["#2a9fe6"]
        )
        st.plotly_chart(fig_future, use_container_width=True)

        # Danger zone
        threshold = acc_df.loc[acc_df["City"] == selected_city, "Threshold (mm)"].values[0]
        max_future = df_future[col].max()

        if max_future > threshold:
            st.error(f"🔴 DANGER: {max_future:.2f} mm exceeds {threshold:.2f} mm!")
        else:
            st.success("🟢 Safe — forecast stays below threshold")

# ------------------------------------------------------
# MODEL COMPARISON PAGE — SF & New Orleans only
# ------------------------------------------------------
elif page == "🔍 Model Comparison":

    st.markdown("<div class='big-title fade-in'>🔍 Multi-Model Comparison</div>", unsafe_allow_html=True)

    lstm_df = safe_read(LSTM_ACC)
    sarima_df = safe_read(SARIMA_ACC)
    xgb_df = safe_read(XGB_ACC)

    if lstm_df is None:
        st.error("LSTM accuracy file missing.")
    else:
        thresholds = dict(zip(lstm_df["City"], lstm_df["Threshold (mm)"]))

        # LSTM
        lstm_df["Model"] = "LSTM"
        lstm_norm = lstm_df[["Model","City","MAE (mm)","RMSE (mm)","Accuracy (%)","Threshold (mm)"]]

        # SARIMA
        if sarima_df is not None:
            sarima_df = sarima_df.rename(columns={"MAE":"MAE (mm)", "RMSE":"RMSE (mm)"})
            sarima_df["Model"] = "SARIMA"
            sarima_df["Threshold (mm)"] = sarima_df["City"].map(thresholds)
            sarima_df["Accuracy (%)"] = (1 - sarima_df["MAE (mm)"]/sarima_df["Threshold (mm)"]) * 100
            sarima_norm = sarima_df[["Model","City","MAE (mm)","RMSE (mm)","Accuracy (%)","Threshold (mm)"]]
        else:
            sarima_norm = pd.DataFrame()

        # XGB
        if xgb_df is not None:
            xgb_df = xgb_df.rename(columns={"MAE":"MAE (mm)", "RMSE":"RMSE (mm)"})
            xgb_df["Model"] = "XGBOOST"
            xgb_df["Threshold (mm)"] = xgb_df["City"].map(thresholds)
            xgb_df["Accuracy (%)"] = (1 - xgb_df["MAE (mm)"]/xgb_df["Threshold (mm)"]) * 100
            xgb_norm = xgb_df[["Model","City","MAE (mm)","RMSE (mm)","Accuracy (%)","Threshold (mm)"]]
        else:
            xgb_norm = pd.DataFrame()

        compare_df = pd.concat([lstm_norm, sarima_norm, xgb_norm], ignore_index=True)

        # Only 2 cities
        city_choice = st.selectbox("Select City", ["San_Francisco_CA", "New_Orleans_LA"])
        city_compare = compare_df[compare_df["City"] == city_choice]

        st.dataframe(city_compare)

        # Accuracy bar
        fig_acc = px.bar(
            city_compare, 
            x="Model", y="Accuracy (%)", color="Model",
            title=f"Accuracy Comparison — {city_choice}",
            text="Accuracy (%)",
            color_discrete_sequence=["#2a9fe6","#2a6bff","#0b4f6c"]
        )
        fig_acc.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_acc, use_container_width=True)

        # MAE/RMSE
        metric_df = city_compare.melt(
            id_vars=["Model"], value_vars=["MAE (mm)", "RMSE (mm)"],
            var_name="Metric", value_name="Value"
        )
        fig_metrics = px.bar(
            metric_df, x="Model", y="Value", color="Metric",
            barmode="group",
            title=f"Error Metrics — {city_choice}",
            color_discrete_sequence=["#7fd6ff", "#2a6bff"]
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown("<div class='footer'>Made for Minor Project — LSTM Sea Level Forecasting System</div>", unsafe_allow_html=True)
