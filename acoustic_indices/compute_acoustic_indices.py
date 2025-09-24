import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from maad import sound, features, util
import librosa
EPS = np.finfo(float).eps
import yaml
import sys
import ast

CONFIG_PATH = 'config_haupert_2025.yaml'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

OUTPUT_PATH = "./output"
DURATION = 60 # sec


def parse_date(date_str):
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%MZ"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format not supported: {date_str}")

def get_file_date(bugg_file_name):

    date = bugg_file_name.replace('.mp3', '')
    date = date.replace('_', ':')
    date = parse_date(date)
    return date

def print_time_information(time_start, i, number_of_files):
    elapsed_time = (time.time() - time_start)
    sec_per_file = elapsed_time/(i+1)
    number_of_remaining_files = number_of_files - i
    remaining_time = (number_of_remaining_files * sec_per_file)/60
    print(f"Processed {i+1}/{number_of_files} files in {elapsed_time/60:.1f} min, {sec_per_file:.1f} sec per file, {remaining_time:.1f} min remaining" , flush=True )


def audio_processing(sig, fs, duration):

    sig = sig[0:duration*fs]

    df_audio_ind = features.all_temporal_alpha_indices(
        sig, fs,
        mode=CONFIG['mode_env'],
        Nt=CONFIG['Nt'],
        gain=CONFIG['gain'],
        sensibility=CONFIG['sensibility'],
        Vadc=CONFIG['sensibility'],
        dt=CONFIG['deltaT'],
        dB_threshold=CONFIG['dB_threshold'],
        rejectDuration=CONFIG['reject_duration'],
        display=False
        )

    Sxx_power, tn, fn, ext = sound.spectrogram(
        sig,fs,
        window=CONFIG['window'],
        nperseg=CONFIG['n_fft'],
        noverlap=CONFIG['hop_length'])


    df_spec_ind, _ = features.all_spectral_alpha_indices(
        Sxx_power,
        tn, fn,
        flim_low=CONFIG['flim_low'],
        flim_mid=CONFIG['flim_mid'],
        flim_hi=CONFIG['flim_hi'],
        R_compatible='soundecology',
        seed_level=CONFIG['seed_level'], 
        low_level=CONFIG['low_level'], 
        fusion_rois=CONFIG['fusion_rois'],
        remove_rois_flim_min = CONFIG['remove_rois_flim_min'],
        remove_rois_flim_max = CONFIG['remove_rois_flim_max'],
        remove_rain = CONFIG['remove_rain'],
        min_event_duration=CONFIG['min_event_duration'], 
        max_event_duration=CONFIG['max_event_duration'], 
        min_freq_bw=CONFIG['min_freq_bw'], 
        max_freq_bw=CONFIG['max_freq_bw'], 
        max_ratio_xy = CONFIG['max_ratio_xy'],
        display=False)

    # --- convert to magnitude spectrogram for CENT / TQ ---
    Sxx_mag = np.sqrt(Sxx_power)
    # --- spectral centroid using librosa ---
    cent_frames = librosa.feature.spectral_centroid(S=Sxx_mag, sr=fs)
    cent = float(np.mean(cent_frames))
    # --- third quartile (TQ / Q75) ---
    mean_mag = np.mean(Sxx_mag, axis=1)
    pmf = mean_mag / np.sum(mean_mag)
    cdf = np.cumsum(pmf)
    tq = float(np.interp(0.75, cdf, fn))

    df_indices = pd.concat([df_audio_ind, df_spec_ind], axis=1)
    df_indices.insert(0, "CENT", [cent])
    df_indices.insert(0, "TQ ", [tq])

    return df_indices




if __name__ == "__main__":

    time_start = time.time()

    df_indices_list = []

    chunk_file = sys.argv[1] 

    with open(chunk_file, "r") as f:

        number_of_files = sum(1 for line in f)
        f.seek(0) # Reset file pointer to the beginning of the file before reading the lines
        print(f"Start processing {number_of_files} files", flush=True)

        
        for i, line in enumerate(f):
            line = line.strip() 

            if line:  # Check if the line is not empty

                try:
                    parts = ast.literal_eval(line)

                    print(parts, flush=True)

                    data_path = parts[0]
                    bugg = parts[1]
                    conf = parts[2]
                    file = parts[3]
                    file_time = get_file_date(file)
                    month_print = parts[4]

                    sig, fs = librosa.load(os.path.join(data_path,bugg,conf,file), mono=True, sr=None)

                    if fs != 44100:
                        print("problem ", file, " fs= ", fs)   
                        
                    df_indices = audio_processing(sig, fs, DURATION)

                    df_indices.insert(loc=0, column='bugg', value=bugg)
                    df_indices.insert(loc=1, column='File', value=file)
                    df_indices.insert(loc=2, column='Date', value=file_time)          
                    df_indices_list.append(df_indices)


                except Exception as e:
                    print(f"Unable to load {line}")
                    print(f"An error occurred: {e}")

            if (i != 0) and (i % 10 == 0):
                print_time_information(time_start, i, number_of_files)

       
    df_spectral_indices = pd.concat(df_indices_list, ignore_index=True)
    df_spectral_indices.to_csv(os.path.join(OUTPUT_PATH,  f"acoustic_indices_{bugg}_{month_print}.csv"))


    print("End processing")
