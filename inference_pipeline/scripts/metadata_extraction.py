#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from pydub import AudioSegment
from mutagen.wave import WAVE
import librosa
import numpy as np

from config import (
    CLIPPED_AUDIO_OUTPUT_FOLDER,
    METADATA_OUTPUT_CSV)


# In[2]:


"""the below function does: 

use cleaned 3 second clips

extracts metadata from :
      filenames, audiomoth metadata, audio properties, extracts acoustic features
      
Save everything into a dataframe in CSV formate"""

def extract_metadata(input_folder, output_csv):
   
    metadata_list = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)

            try:
                # extract metadata from filename
                parts = filename.replace(".wav", "").split('_')
                if len(parts) == 5:
                    recorder_name = '_'.join(parts[:3]) #Audio_Moth_1
                    timestamp = f"{parts[3]}_{parts[4]}" # YYYYMMDD_HHMMSS
                else:
                    print(f"Skipping {filename}: Incorrect filename format")
                    continue

                # extract technical metadata
                audio = WAVE(file_path)
                sample_rate = audio.info.sample_rate  # Hz
                channels = audio.info.channels  # 1/2
                bit_depth = audio.info.bits_per_sample  
                file_size = round(os.path.getsize(file_path) / 1024, 2)  
                duration = round(audio.info.length, 2)  # 3 seconds

                # extract AudioMoth metadata (if available)
                tags = audio.tags
                battery_voltage = tags.get("battery_voltage", "Unknown") if tags else "Unknown"
                gain = tags.get("gain", "Unknown") if tags else "Unknown"

                # extract acoustic features using librosa
                y, sr = librosa.load(file_path, sr=None)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

                # store all above metadata in a dictionary
                metadata_list.append({
                    "clip_name": filename,
                    "Recorder": recorder_name,
                    "Timestamp": timestamp,
                    "Duration (sec)": duration,
                    "Sample Rate (Hz)": sample_rate,
                    "Channels": channels,
                    "Bit Depth": bit_depth,
                    "File Size (KB)": file_size,
                    "Battery Voltage": battery_voltage,
                    "Gain": gain,
                    "Spectral Centroid": spectral_centroid,
                    "Zero Crossing Rate": zero_crossing_rate
                })

                # Little help for logging, showing actual progress
                if len(metadata_list) % 1000 == 0:
                    print('.', end='', flush=True)

            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

    # convert to df and save to CSV
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(output_csv, index=False)

    print(f"\nMetadata saved to {output_csv}")


"""this function will be tested using audio data collected by AudioMoth
"""

if __name__ == "__main__":
    extract_metadata(CLIPPED_AUDIO_OUTPUT_FOLDER, METADATA_OUTPUT_CSV)
    
# In[ ]:




