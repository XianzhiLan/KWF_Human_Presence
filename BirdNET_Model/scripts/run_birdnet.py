# %%
import os
import pandas as pd
import shutil
from tqdm import tqdm
from pathlib import Path
from birdnet import SpeciesPredictions, predict_species_within_audio_file, predict_species_at_location_and_time
from birdnet.models.v2m4 import AudioModelV2M4TFLite
import soundfile as sf
import concurrent.futures
from tqdm import tqdm

# config module not included with this copy; values actually used in production:
# BIRDNET_LATITUDE = 8.33
# BIRDNET_LONGITUDE = -83.3
# BIRDNET_CONFIDENCE_THRESHOLD = 0.25
# (these two coordinates are what predict_species_at_location_and_time() below
# uses to generate the ~218-species regional filter)
from config import (
    CLIPPED_AUDIO_OUTPUT_FOLDER,
    BIRDNET_FINAL_OUTPUT_CSV,
    CORRUPTED_CLIPS_FOLDER_BIRDNET,
    BIRDNET_TIMEOUT_SECONDS,
    LOG_FILE,
    BIRDNET_LATITUDE,
    BIRDNET_LONGITUDE,
    BIRDNET_CONFIDENCE_THRESHOLD
)

# %%
#load the needed model
model = AudioModelV2M4TFLite(language="en_us")



def run_birdnet_on_clip(audio_path, model, species_filter=None, confidence_threshold=0.0):
    audio_path = Path(audio_path)
    clip_name = audio_path.stem
    predictions_dict = {}

    for time_range, species_map in predict_species_within_audio_file(
        audio_path,
        species_filter=species_filter,
        custom_model=model
    ):
        filtered_map = {
            species: conf for species, conf in species_map.items()
            if conf >= confidence_threshold
        }
        predictions_dict[time_range] = filtered_map

    predictions = SpeciesPredictions(predictions_dict)
    rows = []
    for (start, end), species_map in predictions.items():
        if species_map:

            species_arr = []
            conf_arr = []
            for species, conf in species_map.items():
                species_arr.append(species)
                conf_arr.append(conf)

            rows.append({
                "clip_name": f"{clip_name}.wav",
                "start" : start,
                "end" : end,
                "species": species_arr,
                "confidence": conf_arr
            })
        else:
            rows.append({
                "clip_name": f"{clip_name}.wav",
                "start" : start,
                "end" : end,
                "species": [],
                "confidence": []
            })
    return rows



## below function move a problematic clip into the corrupted_clips folder
def move_to_corrupted(audio_path):
    os.makedirs(CORRUPTED_CLIPS_FOLDER_BIRDNET, exist_ok=True)
    shutil.move(audio_path, os.path.join(CORRUPTED_CLIPS_FOLDER_BIRDNET, os.path.basename(audio_path)))

##the below function add time out, prints a message on timeout or other errors
## moves the clip to a corrupted folder. 
def classify_clip_with_timeout(
    audio_path,
    model,
    timeout: float = BIRDNET_TIMEOUT_SECONDS,
    log_file: str = LOG_FILE,
    species_filter=None,
    confidence_threshold=0.0
):
    # Use a single-worker executor so we can timeout cleanly
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            run_birdnet_on_clip,
            audio_path,
            model,
            species_filter,
            confidence_threshold
        )
        try:
            rows = future.result(timeout=timeout)
            return rows, "success"

        except concurrent.futures.TimeoutError:
            msg = f"Timeout: processing {Path(audio_path).name} exceeded {timeout}s"
            print(msg)
            with open(log_file, "a") as log:
                log.write(msg + "\n")
            move_to_corrupted(audio_path)
            return [], "timeout"

        except Exception as e:
            msg = f"Error processing {Path(audio_path).name}: {e}"
            print(msg)
            with open(log_file, "a") as log:
                log.write(msg + "\n")
            move_to_corrupted(audio_path)
            return [], "error"



## below function loops through all 3 seconds clips in a folder 

def process_all_clips(
    clips_folder=CLIPPED_AUDIO_OUTPUT_FOLDER,
    output_csv=BIRDNET_FINAL_OUTPUT_CSV,
    timeout: float = BIRDNET_TIMEOUT_SECONDS,
    log_file: str = LOG_FILE,
    model=model
):
    clips_folder = Path(clips_folder)
    clips = list(clips_folder.glob("*.wav"))
    species_in_area = predict_species_at_location_and_time(BIRDNET_LATITUDE, BIRDNET_LONGITUDE)
    species_filter = set(species_in_area.keys())

    all_rows = []
    success_count = 0
    timeout_count = 0
    error_count = 0

    for clip in tqdm(clips, desc="Processing clips", unit="clip"):

        print(f"Processing {success_count + timeout_count + error_count + 1} / {len(clips)} clips "
          f"({(success_count + timeout_count + error_count + 1) / len(clips) * 100:.1f}%) - Current: {clip.name}")
        
        rows, result = classify_clip_with_timeout(
            audio_path=clip,
            model=model,
            timeout=timeout,
            log_file=log_file,
            species_filter=species_filter,
            confidence_threshold=BIRDNET_CONFIDENCE_THRESHOLD
        )

        if result == "success":
            success_count += 1
            all_rows.extend(rows)
        elif result == "timeout":
            timeout_count += 1
        elif result == "error":
            error_count += 1

    # Save to CSV
    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    print(f"\nSaved final CSV to: {output_csv}")

    # Log summary
    summary = (
        f"\nProcessing complete:\n"
        f"Total clips: {len(clips)}\n"
        f"Successes: {success_count}\n"
        f"Timeouts: {timeout_count}\n"
        f"Errors: {error_count}\n"
    )
    print(summary)
    with open(log_file, "a") as log:
        log.write(summary + "\n")


if __name__ == "__main__":
    process_all_clips()