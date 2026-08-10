# %%
import requests
import soundfile as sf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun

from config import LATITUDE, LONGITUDE, WEATHER_DATA_PATH, AUDIO_MOTH_FOLDER, WEATHER_CODE_LEGEND, TIMEZONE

"""the purpose of this script is to extract hourly weather info for each wav file in each day's folder under Audio Moth 1
     and save one weather CSV file of each wav file. """


## the below function is used to add time of day classification
def get_time_of_day(lat, lon, timestamp_str):
    local_tz = ZoneInfo(TIMEZONE)
    utc_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))
    local_dt = utc_dt.astimezone(local_tz)

    location = LocationInfo(latitude=lat, longitude=lon)
    s = sun(location.observer, date=local_dt.date(), tzinfo=local_tz)

    hour = local_dt.hour
    if 4 <= hour < 6:
        label = "dawn"
    elif 6 <= hour < 12:
        label = "morning"
    elif 12 <= hour < 16:
        label = "afternoon"
    elif 16 <= hour < 19:
        label = "evening"
    else:
        label = "night"

    return {
        "timestamp_local": local_dt,
        "sunrise": s["sunrise"],
        "sunset": s["sunset"],
        "time_of_day": label
    }


# the below function retrieve weather info for one wav file

def get_weather_info(wav_path, lat, lon):
    base_name = wav_path.stem  #  20250317_180003
    


    try:
        start_time = datetime.strptime(base_name, "%Y%m%d_%H%M%S")
        info = sf.info(str(wav_path))
        duration_hours = int(info.frames / info.samplerate // 3600) + 1
        end_time = start_time + timedelta(hours=duration_hours)

        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")

        # making open meteo request

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,windspeed_10m,relative_humidity_2m,weathercode",
            "timezone": "UTC"
        }

        response = requests.get(url, params=params)
        data = response.json()

        #convert API data to a df
        weather_df = pd.DataFrame({
            "datetime": data["hourly"]["time"],
            "temperature": data["hourly"]["temperature_2m"],
            "windspeed": data["hourly"]["windspeed_10m"],
            "precipitation": data["hourly"]["precipitation"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "weathercode": data["hourly"]["weathercode"]
        })


        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

        weather_df["weather_desc"] = weather_df["weathercode"].map(WEATHER_CODE_LEGEND)

        # filter only the hours that fall within the .wav file's duration
        # for example, if the wav file spans only 6 hours, then it keeps only 4 rows, not all 24 hours

        hourly_timestamps = [start_time + timedelta(hours=i) for i in range(duration_hours)]
        hour_set = set([ts.replace(minute=0, second=0, microsecond=0) for ts in hourly_timestamps])
        weather_df = weather_df[weather_df["datetime"].isin(hour_set)]

        ## add time of day features
        time_features = weather_df["datetime"].apply(
            lambda ts: get_time_of_day(lat, lon, ts.strftime("%Y-%m-%d %H:%M:%S"))
        )

        weather_df["timestamp_local"] = time_features.apply(lambda x: x["timestamp_local"])
        weather_df["sunrise"] = time_features.apply(lambda x: x["sunrise"])
        weather_df["sunset"] = time_features.apply(lambda x: x["sunset"])
        weather_df["time_of_day"] = time_features.apply(lambda x: x["time_of_day"])

        return weather_df

    except Exception as e:
        print(f"[ERROR] Failed to process {wav_path.name}: {e}")
        return None

## the below function loop through all the wav file in all 5 days folder 
# call above function on each of the wav file
def get_weather_info_all():
    root_folder = Path(AUDIO_MOTH_FOLDER)
    all_weather_dfs = []

    for day_folder in sorted(root_folder.iterdir()):
        if not day_folder.is_dir():
            continue

        wav_files = list(day_folder.glob("*.WAV"))
        for wav_file in wav_files:
            print(f"Processing: {wav_file.name}")
            weather_df = get_weather_info(wav_file, LATITUDE, LONGITUDE)
            if weather_df is not None:
                all_weather_dfs.append(weather_df)
    if all_weather_dfs:
        combined_df = pd.concat(all_weather_dfs, ignore_index=True)
        combined_df.to_csv(WEATHER_DATA_PATH, index=False)
        print(f"[OK] Combined weather data saved to {WEATHER_DATA_PATH}")
    else:
        print("[WARNING] No weather data collected.")


if __name__ == "__main__":
    get_weather_info_all()


# %%



