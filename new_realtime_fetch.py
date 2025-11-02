# new_realtime_fetch.py
import requests
import pandas as pd
from datetime import datetime, timedelta


def fetch_noaa_historical(station_id, days=365):
    """
    Fetch past 'days' worth of water level data for a NOAA station.
    NOAA API only allows ~30 days per request, so we break into 30-day chunks.
    Data is returned at 6-minute resolution, then resampled to hourly.
    """
    base_url = "https://tidesandcurrents.noaa.gov/api/datagetter"
    end_date = datetime.utcnow()
    begin_date = end_date - timedelta(days=days)

    dfs = []
    chunk_start = begin_date
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=30), end_date)

        params = {
            "product": "water_level",
            "application": "minor_project",
            "begin_date": chunk_start.strftime("%Y%m%d"),
            "end_date": chunk_end.strftime("%Y%m%d"),
            "datum": "MSL",
            "station": station_id,
            "time_zone": "GMT",
            "units": "metric",
            "format": "json"
        }

        r = requests.get(base_url, params=params)
        if r.status_code != 200:
            raise Exception(f"API request failed for {station_id}: {r.status_code}")

        data = r.json().get("data", [])
        if data:
            dfs.append(pd.DataFrame(data))

        # Move to next chunk (avoid overlap)
        chunk_start = chunk_end + timedelta(days=1)

    if not dfs:
        raise Exception(f"No historical data returned for station {station_id}")

    # Combine all chunks
    df = pd.concat(dfs, ignore_index=True)
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df["s"] = pd.to_numeric(df["s"], errors="coerce")

    # Keep only numeric columns & resample to hourly
    numeric_df = df[["t", "v", "s"]].copy()
    numeric_df = numeric_df.set_index("t").resample("1h").mean().reset_index()

    return numeric_df


def main():
    stations = {
        "San_Francisco_CA": "9414290",
        "New_Orleans_LA": "8761724"
    }

    for city, station_id in stations.items():
        try:
            print(f"\n=== Fetching historical data for {city} ===")
            hist_data = fetch_noaa_historical(station_id, days=365)
            filename = f"historical_{city}.csv"
            hist_data.to_csv(filename, index=False)
            print(f"Saved: {filename} ({len(hist_data)} rows)")
        except Exception as e:
            print(f"Failed for {city}: {e}")


if __name__ == "__main__":
    main()
