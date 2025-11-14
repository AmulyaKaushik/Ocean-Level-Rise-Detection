import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Ocean Level Rise – LSTM Forecast Dashboard",
    layout="wide"
)

st.title("🌊 Ocean Level Rise – LSTM Forecast Dashboard")
st.markdown("Interactive LSTM-based forecasts for 9 U.S. coastal cities.")

# -------------------------------------------------------
# Fixed list of 9 cities
# -------------------------------------------------------
CITIES = [
    "San_Francisco_CA",
    "Los_Angeles_CA",
    "Seattle_WA",
    "Miami_FL",
    "New_York_NY",
    "Boston_MA",
    "New_Orleans_LA",
    "Galveston_TX",
    "Honolulu_HI"
]

# Sidebar controls
st.sidebar.header("Controls")
selected_city = st.sidebar.selectbox("Select City", CITIES)

# -------------------------------------------------------
# Load LSTM forecast files
# -------------------------------------------------------
hist_file = f"forecast_{selected_city}_LSTM.csv"
future_file = f"forecast_future_{selected_city}_LSTM.csv"
accuracy_file = "model_accuracy_LSTM.csv"

df_hist = None
df_future = None
acc_df = None

if os.path.exists(hist_file):
    df_hist = pd.read_csv(hist_file)
else:
    st.error(f"Missing file: {hist_file}")

if os.path.exists(future_file):
    df_future = pd.read_csv(future_file)

if os.path.exists(accuracy_file):
    acc_df = pd.read_csv(accuracy_file)
else:
    st.warning("Accuracy file missing: model_accuracy_LSTM.csv")

# -------------------------------------------------------
# Accuracy Table (LSTM only)
# -------------------------------------------------------
st.subheader("📊 LSTM Model Accuracy")
if acc_df is not None:
    st.dataframe(acc_df)
else:
    st.info("LSTM accuracy table not available.")

# -------------------------------------------------------
# Historical Prediction Plot
# -------------------------------------------------------
if df_hist is not None:
    st.subheader(f"📉 Historical Evaluation – {selected_city} (LSTM)")

    fig1 = px.line(
        df_hist,
        y=["Actual (mm)", "Predicted (mm)"],
        title=f"{selected_city} – LSTM Historical Performance",
        labels={"value": "Water Level (mm)", "index": "Time Step"}
    )
    st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------------
# Future Forecast Plot
# -------------------------------------------------------
if df_future is not None:
    st.subheader(f"🔮 Future Forecast – {selected_city} (LSTM)")

    fig2 = px.line(
        df_future,
        x="Date",
        y="Forecasted Value (mm)",
        title=f"{selected_city} – Next Forecasted Days (LSTM)",
        labels={"Forecasted Value (mm)": "Water Level (mm)"}
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No future forecast available for this city.")

# -------------------------------------------------------
# Compare All Cities – LSTM Only
# -------------------------------------------------------
st.subheader("🌍 Compare All Cities (LSTM)")

compare_dfs = []

for city in CITIES:
    fpath = f"forecast_{city}_LSTM.csv"
    if os.path.exists(fpath):
        d = pd.read_csv(fpath)
        d["City"] = city
        d["Step"] = range(len(d))
        compare_dfs.append(d)

if compare_dfs:
    combined = pd.concat(compare_dfs)

    fig3 = px.line(
        combined,
        x="Step",
        y="Predicted (mm)",
        color="City",
        title="LSTM Prediction Comparison Across Cities",
        labels={"Predicted (mm)": "Water Level (mm)"}
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Comparison data missing for multiple cities.")

st.markdown("---")
st.markdown("Made for Minor Project – LSTM Sea Level Forecasting System")
