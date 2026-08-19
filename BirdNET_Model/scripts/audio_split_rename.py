#!/usr/bin/env python
# coding: utf-8

import os
import datetime
import numpy as np
import soundfile as sf
import librosa
import config


"""Splits hours-long AudioMoth recordings into exactly 3-second clips and renames them.

Each clip is guaranteed to be exactly config.SAMPLES_PER_CLIP samples (144000 at 48kHz),
ensuring sample-accurate alignment with the BirdNET reference CSV.
If a source file's sample rate differs from TARGET_SAMPLE_RATE, it is resampled first.
Incomplete tail clips (shorter than a full 3s) are discarded.

Output filename format: recorderID_YYYYMMDD_HHMMSS.wav
"""


def split_audio(input_folder, output_folder):
    folder_name = os.path.basename(input_folder)
    recorder_id = '_'.join(folder_name.split('_')[:-1])  # e.g. "Audio_Moth_1"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".wav") or filename.lower().startswith("."):
            continue

        audio_path = os.path.join(input_folder, filename)

        base_name = filename.split('.')[0]
        try:
            original_datetime = datetime.datetime.strptime(base_name, "%Y%m%d_%H%M%S")
        except ValueError:
            print(f"Skipping {filename}: Invalid timestamp format")
            continue

        
        data, sr = sf.read(audio_path)

        # Resample to target rate if the recorder was configured differently
        if sr != config.TARGET_SAMPLE_RATE:
            if data.ndim > 1:
                # soundfile: (samples, channels) → librosa wants (channels, samples)
                data = librosa.resample(data.T, orig_sr=sr, target_sr=config.TARGET_SAMPLE_RATE).T
            else:
                data = librosa.resample(data, orig_sr=sr, target_sr=config.TARGET_SAMPLE_RATE)
            sr = config.TARGET_SAMPLE_RATE

        n = config.SAMPLES_PER_CLIP
        total_samples = len(data)

        clip_index = 0
        for start in range(0, total_samples - n + 1, n):
            chunk = data[start:start + n]
            new_timestamp = original_datetime + datetime.timedelta(seconds=clip_index * 3)
            new_time_str = new_timestamp.strftime('%Y%m%d_%H%M%S')
            clip_filename = f"{recorder_id}_{new_time_str}.wav"
            sf.write(os.path.join(output_folder, clip_filename), chunk, sr)
            clip_index += 1

        print(f"Processed {filename}: {clip_index} clips")


def split_all_audio(main_folder, output_folder):
    for folder in os.listdir(main_folder):
        path = os.path.join(main_folder, folder)
        if not os.path.isdir(path):
            continue
        split_audio(input_folder=path, output_folder=output_folder)


if __name__ == "__main__":
    split_all_audio(
        main_folder=config.RAW_AUDIO_INPUT_FOLDER,
        output_folder=config.CLIPPED_AUDIO_OUTPUT_FOLDER
    )


"""Folder layout expected under RAW_AUDIO_INPUT_FOLDER:

raw/
  Audio_Moth_1_032125/
    20250321_062544.wav
    20250321_180002.wav
  Audio_Moth_2_032025/
    20250320_062541.wav

Example output clips:
  Audio_Moth_1_20250321_062544.wav  (exactly 144000 samples)
  Audio_Moth_1_20250321_062547.wav  (exactly 144000 samples)
  ...
"""
