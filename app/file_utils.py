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


def select_samples_from_recordings(filters, csv_file, padding, export_path, dataset_path ) :

    export_df = pd.read_csv(os.path.join(export_path, csv_file))
    export_folder = csv_file.split(".")[0]
    os.makedirs(os.path.join(export_path, export_folder  ))
    shutil.copyfile(os.path.join(export_path, csv_file), os.path.join(export_path, export_folder, csv_file))

    for index, row in export_df.iterrows():

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
                output_mp3 = os.path.join(export_path, export_folder  , row["Sample_filename"] )
                audio_segment = AudioSegment.from_wav(temp_wav)
                audio_segment.export(output_mp3, format="mp3")

                # Remove temp file
                os.remove(temp_wav)
                print(recording_path, "split and exported")

            else:
                print(recording_path, "does not exist.")


    # export metadata test
    metadata_path = os.path.join(export_path, export_folder, "export_metadata.json")
    with open(metadata_path, "w") as outfile:
        json.dump(filters.model_dump(), outfile, indent=2)

    output_mp3 = os.path.join(export_path, export_folder  , row["Sample_filename"] )

    shutil.make_archive(os.path.join(export_path, export_folder  ), 'zip', os.path.join(export_path, export_folder ))
    shutil.rmtree(os.path.join(export_path, export_folder  ))
    os.remove( os.path.join(export_path, csv_file) )

    zip_path = f"{os.path.join(export_path, export_folder)}.zip"

    return zip_path