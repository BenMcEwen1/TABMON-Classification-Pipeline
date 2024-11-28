import os.path
import librosa
from multiprocessing import Pool
import soundfile as sf
import pandas as pd
import mgrs
from typing import Any
import csv

RANDOM_SEED = 1337
SAMPLE_RATE_AST = 32000
SAMPLE_RATE = 48000
SIGNAL_LENGTH = 3 # seconds
SPEC_SHAPE = (298, 128) # width x height
FMIN = 20
FMAX = 15000
MAX_AUDIO_FILES = 10000


def save_chunk(args):
    #Function to save a single chunk of audio data to a file.
    chunk, save_path, rate = args
    sf.write(save_path, chunk, rate)

def split_signals(filepath, output_dir, signal_length=15, n_processes=None):
    # files = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]

    # for file in filepath:
    file = filepath # Already looping through files
    sig, rate = librosa.load(file, sr=None)
    sig_splits = [sig[i:i + int(signal_length * rate)] for i in range(0, len(sig), int(signal_length * rate)) if len(sig[i:i + int(signal_length * rate)]) == int(signal_length * rate)]

    with Pool(processes=n_processes) as pool:
        args_list = []
        for s_cnt, chunk in enumerate(sig_splits):
            save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file))[0]}_{s_cnt}.wav")
            args_list.append((chunk, save_path, rate))
        
        # Save each chunk in parallel
        pool.map(save_chunk, args_list)

def load_species_list(path):
    # Load the species list from a file.
    species_list = []
    with open(path) as file:
        for line in file:
            species_list.append(line.strip())
    return sorted(species_list)

def get_species_list(recording_lat, recording_long):
    "Concert WGS84 lat-lon to MGRS UTM coordinates"

    # Load occurances data from csv file
    file_path = "inputs/ebba2_data_occurrence_50km.csv"
    df = pd.read_csv(file_path, delimiter=';')
    
    m = mgrs.MGRS()
    c = m.toMGRS(recording_lat, recording_long)

    input_code = c[0:5] 

    # Filter the DataFrame based on the input code
    filtered_df = df[df['cell50x50'].str.startswith(input_code)]

    # Get the list of unique bird species from the filtered DataFrame
    species_list = pd.DataFrame(filtered_df['birdlife_scientific_name'].drop_duplicates())
    #species_list.loc[len(species_list), 'birdlife_scientific_name'] = 'Noise'
    #species_list.to_csv('species_list.csv', index=False, header=False)
    
    return species_list

def setup_filtering(lat, lon, add_filtering, flist, species_list):
    """Setup filtering based on geographic location."""
    if lat != None and lon != None:
        filtering_list_series = get_species_list(lat, lon)
        filtering_list = filtering_list_series['birdlife_scientific_name'].tolist()
    elif not add_filtering:
        filtering_list = None
    else: 
        filtering_list = []
        with open(flist) as f:
            for line in f:
                filtering_list.append(line.strip())
                
    return filtering_list 


# Output results to json

def add_predictions(
    region_group_id: str,
    media_id: str,
    begin_time: float,
    end_time: float,
    scientific_names: list[str],
    probabilities: list[float],
    uncertainty: float,
    analysis_results: dict[str, Any]
) -> None:
    # Predictions template
    assert len(scientific_names) == len(probabilities)

    if scientific_names:
        analysis_results["region_groups"].append(
            {
                "id": region_group_id,
                "regions": [
                    {
                        "media_id": media_id,
                        "box": {
                            "t1": float(begin_time),
                            "t2": float(end_time)
                        }
                    }
                ]
            }
        )

    for prediction_index, scientific_name in enumerate(scientific_names):
        # print(uncertainty)
        analysis_results["predictions"].append(
            {
                "region_group_id": region_group_id,
                "taxa": {
                    "type": "multilabel",
                    "items": [
                        {
                            "scientific_name": scientific_name,
                            "probability": probabilities[prediction_index],
                            "uncertainty": uncertainty,
                            # Fill in taxon_id if available: "taxon_id": "",
                        }
                    ]
                }
            }
        )

def format_time(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    # Format seconds with leading zero if necessary
    return f"{minutes}:{seconds:02d}"

def create_json(analysis_results, predictions, scores, uncertainty, files, flist, df, add_csv, fname, m_conf, filtered=False):
     # Create a predictions file name
    filename_without_ext = fname.split('.')[0]  
    filename = filename_without_ext.split('\\')[-1]
    pred_name = './pipeline/outputs/predictions_' + filename

    if add_csv:
        with open(f'{pred_name}.csv', 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(["Begin Time", "End Time", "File", "Prediction", "Score"])  # write header

    # Include the input directory when running in directories mode and args.i is referring to a directory.
    file_path = f'{flist}/{fname}' # if algorithm_mode == AlgorithmMode.DIRECTORIES and os.path.isdir(args.i) else fname

    # Add each file to the 'media' list in analysis_results
    analysis_results['media'].append({"filename": file_path, "id": fname})
    
    for i, prediction in enumerate(predictions):
        prediction_sp = []
        begin_time = i * 3
        end_time = begin_time + 3
        formatted_begin_time = format_time(begin_time)
        formatted_end_time = format_time(end_time)
        
        # Set a threshold for scores, 0.1 for unfiltered and 0.2 for filtered
        threshold = m_conf
        for name, score in zip(prediction, scores[i]):

            row = df[df['ScientificName'] == name]
            # Extract the ScientificName from the matched row
            common_name = row['CommonName'].values[0] if not row.empty else 'Not found'
            prediction_sp.append(common_name)
            
            if add_csv:
                with open(f'{pred_name}.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    #writer.writerow([begin_time, end_time, files[i], name, score]) # uncomment for time in seconds in csv
                    writer.writerow([formatted_begin_time, formatted_end_time, fname, f'{common_name}_{name}', score]) # uncomment for time in minutes:seconds in csv
        
        region_group_id = f"{file_path}?region={i}"
        add_predictions(region_group_id, fname, begin_time, end_time, prediction, scores[i], uncertainty[i], analysis_results)


def create_json_maxpool(analysis_results, predictions, scores, uncertainty, files, flist, df, add_csv, fname, m_conf, length, filtered=False):
    # Create a predictions file name
    filename_without_ext = fname.split('.')[0]  
    pred_name = 'outputs/predictions_' + filename_without_ext

    if add_csv:
        with open(f'{pred_name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File", "Prediction", "Score"])  # write header
    
    begin_time = 0
    end_time = length*3

    # Include the input directory when running in directories mode and args.i is referring to a directory.
    file_path = f'{flist}/{fname}' # if algorithm_mode == AlgorithmMode.DIRECTORIES and os.path.isdir(args.i) else fname

    # Add each file to the 'media' list in analysis_results
    analysis_results['media'].append({"filename": file_path, "id": fname})
    prediction_sp = []
        
    # Set a threshold for scores, 0.1 for unfiltered and 0.2 for filtered
    threshold = m_conf
    for name, score in zip(predictions, scores):
        print(name, score)

        row = df[df['ScientificName'] == name]
        # Extract the ScientificName from the matched row
        common_name = row['CommonName'].values[0] if not row.empty else 'Not found'
        prediction_sp.append(common_name)
        
        if add_csv:
            with open(f'{pred_name}.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                #writer.writerow([begin_time, end_time, files[i], name, score]) # uncomment for time in seconds in csv
                writer.writerow([fname, common_name, score]) # uncomment for time in minutes:seconds in csv
    
    region_group_id = f"{file_path}?region={0}"

    # add_predictions(region_group_id, fname, begin_time, end_time, predictions, scores, uncertainty[i], analysis_results)