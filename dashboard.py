import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="Ocean Level Rise – LSTM Dashboard", layout="wide")
st.title("🌊 Ocean Level Rise – LSTM Forecast Dashboard")
st.markdown("Interactive LSTM forecasts with danger-zone alert and tide station visualization.")

# ----------------------------
# Fixed cities + NOAA coords
# ----------------------------
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

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")
selected_city = st.sidebar.selectbox("Select City", CITIES)
show_map = st.sidebar.checkbox("Show Tide Stations Map", value=True)

# ----------------------------
# File paths
# ----------------------------
hist_file = f"forecast_{selected_city}_LSTM.csv"
future_file = f"forecast_future_{selected_city}_LSTM.csv"
accuracy_file = "model_accuracy_LSTM.csv"

def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None

df_hist = safe_read(hist_file)
df_future = safe_read(future_file)
acc_df = safe_read(accuracy_file)

# ----------------------------
# Accuracy table
# ----------------------------
st.subheader("📊 LSTM Model Accuracy")
if acc_df is not None:
    st.dataframe(acc_df)
else:
    st.warning("model_accuracy_LSTM.csv not found!")

# ----------------------------
# Tide station map
# ----------------------------
if show_map:
    st.subheader("📍 Tide Stations Map")

    m = folium.Map(location=[37, -97], zoom_start=4, tiles="CartoDB positron")
    for city, (lat, lon) in CITY_COORDS.items():
        folium.Marker(
            [lat, lon],
            tooltip=city,
            popup=city.replace("_", " ")
        ).add_to(m)

    st_folium(m, width=700, height=400)

# ----------------------------
# Historical Evaluation Plot (No anomalies)
# ----------------------------
st.subheader(f"📉 Historical Evaluation – {selected_city} (LSTM)")

if df_hist is None:
    st.error(f"Missing: {hist_file}")
else:
    df_hist = df_hist.reset_index(drop=True)
    df_hist["Step"] = df_hist.index

    if {"Actual (mm)", "Predicted (mm)"}.issubset(df_hist.columns):
        fig_hist = px.line(
            df_hist,
            x="Step",
            y=["Actual (mm)", "Predicted (mm)"],
            title=f"{selected_city} – Historical Forecast (LSTM)",
            labels={"value": "Water Level (mm)"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.error(f"{hist_file} missing required columns.")

# ----------------------------
# Future Forecast Plot + Danger Zone Alert
# ----------------------------
st.subheader(f"🔮 Future Forecast – {selected_city} (LSTM)")

if df_future is None:
    st.info("No future forecast available.")
else:
    # Plot
    if "Date" in df_future.columns:
        value_col = [c for c in df_future.columns if "Forecasted" in c][0]
        fig_future = px.line(
            df_future,
            x="Date",
            y=value_col,
            title=f"{selected_city} – Future Forecast (LSTM)",
            labels={value_col: "Water Level (mm)"}
        )
        st.plotly_chart(fig_future, use_container_width=True)

        # ----------------------------
        # 🔴 DANGER ZONE ALERT
        # ----------------------------
        if acc_df is not None:
            thr = acc_df.loc[acc_df["City"] == selected_city, "Threshold (mm)"]
            if not thr.empty:
                threshold = float(thr.iloc[0])
                max_future = df_future[value_col].max()

                if max_future > threshold:
                    st.error(
                        f"🔴 **DANGER ZONE ALERT:** "
                        f"Forecasted water level **{max_future:.2f} mm** "
                        f"exceeds threshold **{threshold:.2f} mm**!"
                    )
                else:
                    st.success("✅ Forecast stays below danger threshold.")
    else:
        st.error(f"{future_file} is missing a Date column.")

# ----------------------------
# Compare all cities (LSTM only)
# ----------------------------
st.subheader("🌍 Compare All Cities (LSTM Predictions)")

combined = []
for city in CITIES:
    fp = f"forecast_{city}_LSTM.csv"
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        if "Predicted (mm)" in d.columns:
            d["City"] = city
            d["Step"] = range(len(d))
            combined.append(d[["Step", "Predicted (mm)", "City"]])

if combined:
    df_all = pd.concat(combined)
    fig_compare = px.line(
        df_all,
        x="Step",
        y="Predicted (mm)",
        color="City",
        title="LSTM Predictions Comparison Across Cities",
        labels={"Predicted (mm)": "Water Level (mm)"}
    )
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.warning("No LSTM forecast files found for comparison.")

st.markdown("---")
st.markdown("Made for Minor Project – LSTM Sea Level Forecasting System")
