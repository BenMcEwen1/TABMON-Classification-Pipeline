import os
import zipfile
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import static_ffmpeg
static_ffmpeg.add_paths()
import librosa
import shutil
import soundfile as sf
from pydub import AudioSegment
import json
from tqdm import tqdm
import time
from collections import Counter
import matplotlib.pyplot as plt
from math import log
import math
import random


def find_audio_file(segment_row, audio_dir="./audio/segments/"):
    if not os.path.exists(audio_dir):
        return None
        
    #files_map = {f.lower(): f for f in os.listdir(audio_dir)}
    
    possible_names = [
        segment_row['filename'],
        segment_row['filename'].lower(),
    ]
    
    if 'audio_filename' in segment_row and 'device_id' in segment_row and 'start_time' in segment_row:
        base_filename = os.path.splitext(segment_row['audio_filename'])[0]
        segment_start = int(segment_row['start_time'])
        index = int(segment_start / 3)
        transformed = f"{base_filename}_{segment_row['device_id']}_{index}.wav"
        possible_names.append(transformed)
        possible_names.append(transformed.lower())
    
    for name in possible_names:
        path = os.path.join(audio_dir, name)
        if os.path.exists(path):
            return path

        # Case-insensitive match
        #if name.lower() in files_map:
        #return os.path.join(audio_dir, files_map[name.lower()])
            
    return None

def find_embedding_file(segment_row, embedding_dir="./audio/embeddings/"):
    if not os.path.exists(embedding_dir):
        return None
        
    # Get all files in directory with case-insensitive lookup
    #files_map = {f.lower(): f for f in os.listdir(embedding_dir)}
    
    # Try different filename formats
    possible_names = []
    
    # Format 1: Basic approach - just change extension
    base_name = os.path.splitext(segment_row['filename'])[0].lower()
    possible_names.append(f"{base_name}.pt")
    
    # Format 2: Try with device ID and index
    if 'audio_filename' in segment_row and 'device_id' in segment_row and 'start_time' in segment_row:
        base_filename = os.path.splitext(segment_row['audio_filename'])[0]
        segment_start = int(segment_row['start_time'])
        index = int(segment_start / 3)
        transformed = f"{base_filename.lower()}_{segment_row['device_id']}_{index}.pt"
        possible_names.append(transformed)
    
    # Try each name
    for name in possible_names:
        # Direct match
        path = os.path.join(embedding_dir, name)
        if os.path.exists(path):
            return path
            
        # Case-insensitive match
        #if name.lower() in files_map:
        #    return os.path.join(embedding_dir, files_map[name.lower()])
            
    return None

def create_zip_archive(files, prefix="export"):
    if not files:
        return None
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name
        
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            file_path = Path(file)
            if file_path.exists():
                zipf.write(file_path, arcname=file_path.name)
                
    return zip_path


country_to_folder = {
    "France": "proj_tabmon_NINA_FR",
    "Norway": "proj_tabmon_NINA",
    "Netherlands": "proj_tabmon_NINA_NL",
    "Spain": "proj_tabmon_NINA_ES"
}

def bugg_id_to_folder(id_str):
    # Ensure the final ID is 16 characters long by left-padding with zeros
    padded_id = id_str.rjust(15, '0')
    return f"bugg_RPiID-1{padded_id}"


def audio_split(sig, start_time, rate, padding):
    sample_length = 3 #sec
    sig_padded = np.concatenate((np.zeros(rate*padding), sig, np.zeros(rate*padding ) ) )

    sample = sig_padded[ start_time*rate :  start_time*rate + sample_length*rate + 2* padding*rate ]

    ## Put the boundaries of the sample to 0 to vizualize it on the sepctrogram 
    sample[ padding*rate - 1024 : padding*rate ] = 0
    sample[ padding*rate + sample_length*rate : padding*rate + sample_length*rate + 1024 ] = 0
    return sample


def clean(x):
    return [species.strip() for species in x.split(",")]

def stratify_balanced_with_min(data, min_samples=10, max_samples=100, metric="kl"):    
    # Count total available species
    species_list = []
    for _, row in data.iterrows():
        species_list.extend(clean(row["Predictions"]))
    species_counts = Counter(species_list)
    species_collected = {s: 0 for s in species_counts}
    k = len(species_collected)

    def imbalance_variance(counts, target):
        return sum((counts[s] - target)**2 for s in counts)

    def imbalance_kl(counts):
        total = sum(counts.values())
        if total == 0:
            return float("inf")
        q = 1.0 / k
        kl = 0.0
        for s, c in counts.items():
            p = c / total
            if p > 0:
                kl += p * math.log(p / q)
        return kl

    selected_indices = set()

    while len(selected_indices) < max_samples:
        best_row, best_score = None, float("inf")

        for idx, row in data.iterrows():
            if idx in selected_indices:
                continue
            preds = clean(row["Predictions"])

            # Simulate adding this row
            temp_counts = species_collected.copy()
            for p in preds:
                if p in temp_counts:
                    temp_counts[p] += 1

            # Are there species still under min_samples?
            under_min = [s for s, c in species_collected.items() if c < min_samples]

            if under_min:
                # Require that row covers at least one underrepresented species
                if not any(p in under_min for p in preds):
                    continue  # row not useful for meeting min constraint

            # Evaluate imbalance
            if metric == "variance":
                target = (len(selected_indices)+1)/k
                score = imbalance_variance(temp_counts, target)
            elif metric == "kl":
                score = imbalance_kl(temp_counts)
            else:
                raise ValueError("Unknown metric")

            if score < best_score + 1e-5:
                best_score = score
                best_row = idx

        if best_row is None:
            # pick any remaining candidate randomly
            remaining = [idx for idx in data.index if idx not in selected_indices]
            if not remaining:
                break  # Nothing left
            best_row = random.choice(remaining)

        # Commit best row
        selected_indices.add(best_row)
        for p in clean(data.loc[best_row, "Predictions"]):
            if p in species_collected:
                species_collected[p] += 1

        # Early exit: if all species >= min_samples and budget hit, done
        if all(c >= min_samples for c in species_collected.values()) and len(selected_indices) >= max_samples:
            break

    return data.loc[list(selected_indices)]


def select_samples_from_recordings(filters, export_df, padding, export_path, dataset_path) :
    step = 2
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_file = f"export_{timestamp}.csv"

    if filters.stratified:
        print(f"[Step {step}] Stratifying results...", end="")
        start = time.time()
        export_df = stratify_balanced_with_min(export_df, min_samples=2, max_samples=filters.query_limit)
        print(f" Complete [{(time.time() - start):.2f} s]")
        step += 1

    export_df.to_csv(os.path.join(export_path, csv_file), index=False)

    print(f"[Step {step}] Collecting audio segments...")
    export_folder = csv_file.split(".")[0]
    os.makedirs(os.path.join(export_path, export_folder  ))
    shutil.copyfile(os.path.join(export_path, csv_file), os.path.join(export_path, export_folder, csv_file))

    for index, row in tqdm(export_df.iterrows(), total=export_df.shape[0]):
        country_folder  = country_to_folder[row["Country"]]
        bugg_folder = bugg_id_to_folder(row["Device_id"])

        bugg_path = os.path.join(dataset_path, country_folder, bugg_folder)
        conf_folder_list =  [f for f in os.listdir(bugg_path) if os.path.isdir(os.path.join(bugg_path, f))]

        for conf_folder in conf_folder_list:
            
            recording_path = os.path.join(bugg_path, conf_folder, row["Filename"])

            if os.path.exists(recording_path):

                sig, rate = librosa.load(recording_path, mono=True, sr=None)
                start_time = row["Start_time"]

                sample = audio_split(sig, start_time, rate, padding)
                
                # Save to temporary WAV
                temp_wav = os.path.join(export_path, export_folder , "temp.wav")
                sf.write(temp_wav, sample, rate)

                # Convert WAV to MP3 
                output_mp3 = os.path.join(export_path, export_folder, row["Sample_filename"] )
                audio_segment = AudioSegment.from_wav(temp_wav)
                audio_segment.export(output_mp3, format="mp3")

                # Remove temp file
                os.remove(temp_wav)
            else:
                print(recording_path, "does not exist.")
    step += 1

    print(f"[Step {step}] Exporting metadata...", end="")
    start = time.time()
    # export metadata test
    metadata_path = os.path.join(export_path, export_folder, "export_metadata.json")
    with open(metadata_path, "w") as outfile:
        json.dump(filters.model_dump(), outfile, indent=2)

    output_mp3 = os.path.join(export_path, export_folder  , row["Sample_filename"] )

    shutil.make_archive(os.path.join(export_path, export_folder  ), 'zip', os.path.join(export_path, export_folder ))
    shutil.rmtree(os.path.join(export_path, export_folder  ))
    os.remove( os.path.join(export_path, csv_file) )

    zip_path = f"{os.path.join(export_path, export_folder)}.zip"
    print(f" Complete [{(time.time() - start):.2f} s]")

    return zip_path