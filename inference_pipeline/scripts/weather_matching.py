import os
import pandas as pd
from datetime import datetime
import config

# Turns YYYYMMDD_HHMMSS to workable time data
def parse_timestamp(timestamp_str):
    # Read into datetime
    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    if not dt:
        raise ValueError(f"Date Time {timestamp_str} not parsable")

    # Convert to useful representation, (float, seconds since epoch)
    return dt

def round_to_hour(timestamp):
    # Get timestamp (from the metadata) and parse
    dt = parse_timestamp(timestamp)

    # Return timestamp in format of weather data
    return pd.to_datetime(dt).round('h').strftime("%Y-%m-%d %H:%M:%S")

def add_weather_data(df_metadata, df_weather):
    # First round the df_metadata timestamps to nearest hour (aligns with weather data)
    df_metadata['Datetime'] = df_metadata['Timestamp'].apply(round_to_hour)

    # Merge metadata and weather so all weather is applied to specific times
    df_merged = df_metadata.merge(df_weather, on='Datetime', how='left')

    # Return just the clip names, and relevant weather data
    cols = df_weather.columns.tolist()
    cols.append('clip_name')

    return df_merged[cols]

def weather_match():
    # Load weather data, metadata
    df_weather = pd.read_csv(config.WEATHER_DATA_PATH)
    df_metadata = pd.read_csv(config.METADATA_OUTPUT_CSV)

    # Get matches, save
    df_matches = add_weather_data(df_metadata, df_weather)
    df_matches.to_csv(config.WEATHER_OUTPUT_CSV, index=False)

if __name__ == "__main__":
    weather_match()