## generate sbatch and chunk files for recordings tha#  have been missed 
## for example, recordings that have been uploaded after the month has been processed

import os
import math
import pandas as pd
import numpy as np
import shutil
import random
from datetime import datetime
import ast

random.seed(11)
today_date = datetime.today().strftime('%Y-%m-%d')

INDEX_PATH = "/DYNI/tabmon/tabmon_data/index.parquet"

MONTH_SELECTION = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10"]

DATASET_PATH = "/DYNI/tabmon/tabmon_data" 
PIPE_LINE_PATH = "./"
SBATCH_OUTPUT_FILE = "inference_missed_files.sh"
PYTHON_SCRIPT = "inference_parallel.py" 

OUTPUT_CHUNK_FOLDER = f"chunk_files_missed_at_{today_date}"

META_DATA_PATH = "site_info.csv"
META_DATA_DF = pd.read_csv(os.path.join(DATASET_PATH, META_DATA_PATH) , encoding='utf-8').fillna("")


COUNTRY_TO_FOLDER = {
    "Norway": "proj_tabmon_NINA",
    "Spain": "proj_tabmon_NINA_ES",
    "France": "proj_tabmon_NINA_FR",
    "Netherlands": "proj_tabmon_NINA_NL"
}


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

def get_file_year_month(bugg_file_name):
    """Get file year-month"""
    fulldate = bugg_file_name.split("-")
    year_month = fulldate[0] + '-' + fulldate[1]
    return year_month


def get_stored_files(index_path, month_selection):
    index_df = pd.read_parquet(index_path, engine='pyarrow')
    index_df = index_df[index_df["MimeType"]=='audio/mpeg']
    index_df = index_df[list(map(lambda x: x.startswith('bugg_RPiID'), index_df['device']))]

    index_df["month"] = index_df.apply(lambda row: get_file_year_month(row['Name']), axis=1)
    index_df = index_df[index_df["month"].isin(month_selection)]
    all_stored_files = list(map(lambda x: os.path.join(DATASET_PATH,x), index_df['Path']))

    #all_stored_files = index_df["Path"].tolist()
    return all_stored_files


def get_analyzed_files(chunk_path):

    chunk_folder_list = [f for f in os.listdir(chunk_path) if f.startswith("chunk_files") and os.path.isdir(f)]

    print(chunk_folder_list)
    
    already_analyzed_files = []

    for chunk_folder in chunk_folder_list:
        chunk_file_list = [f for f in os.listdir(chunk_folder) if f.startswith("file_chunks") and f.endswith(".txt")]


        for chunk_file in chunk_file_list :
            with open(os.path.join(chunk_folder, chunk_file), encoding='utf-8') as file:
                chunk_lines = file.readlines()
                
            for line in chunk_lines: 
                parts = ast.literal_eval(line)
                recording_path = os.path.join(parts[0], parts[1],parts[2],parts[3], parts[4] )
                already_analyzed_files.append(recording_path)

        print(chunk_folder, len(chunk_file_list), len(chunk_lines), len(already_analyzed_files))

    already_analyzed_files = list(set(already_analyzed_files))

    return already_analyzed_files


def deployment_filter(file_path, bugg_ID, deployment_start, deployment_end):
    spath = os.path.normpath(file_path).split(os.sep)
    #print(spath)
    bugg_filter = spath[5].endswith(bugg_ID)    
    file_date = get_file_date(spath[7])
    date_filter = file_date > deployment_start and file_date < deployment_end

    return bugg_filter and date_filter

def format_metadata( file_path, deploymentID, country, cluster_name, site_name, lat, long ):

    spath = os.path.normpath(file_path).split(os.sep)
    #data = [DATASET_PATH, country_folder, bugg_folder, conf_folder, file, deploymentID, country, cluster_name, site_name, float(lat), float(long)]

    return [os.path.join("/", spath[0], spath[1], spath[2], spath[3]) , spath[4], spath[5], spath[6], spath[7], deploymentID, country, cluster_name, site_name, float(lat), float(long)]




if __name__ == "__main__":

    all_stored_files = get_stored_files(INDEX_PATH, MONTH_SELECTION)
    all_stored_files = set(all_stored_files)
    
    already_analyzed_files = get_analyzed_files(PIPE_LINE_PATH)
    already_analyzed_files = set(already_analyzed_files)

    missed_files = list(all_stored_files - already_analyzed_files)

    print(missed_files[0:2])

    print( len(all_stored_files), " - ", len(already_analyzed_files), " = ", len(missed_files) ) 

    files_data = []

    for index, meta_data_row in META_DATA_DF.iterrows():

        missed_count = 0
        analyzed_count = 0

        deploymentID = meta_data_row["DeploymentID"]
        bugg_ID = meta_data_row["DeviceID"]
        country = meta_data_row["Country"]
        cluster_name = meta_data_row["Cluster"]
        site_name = meta_data_row["Site"]
        lat = meta_data_row["Latitude"]
        long = meta_data_row["Longitude"]
        start_date = meta_data_row["deploymentBeginDate"]
        start_time = meta_data_row["deploymentBeginTime"]
        end_date = meta_data_row["deploymentEndDate"]
        end_time = meta_data_row["deploymentEndTime"]

        deployment_start = datetime.strptime(f"{start_date} {start_time}", "%d/%m/%Y %H:%M:%S")

        if end_date == "" and end_time == "" :
            deployment_end = datetime(3000, 1, 1)
        else :
            deployment_end = datetime.strptime(f"{end_date} {end_time}", "%d/%m/%Y %H:%M:%S")

        

        filtered_missed_files = [f for f in missed_files if deployment_filter(f, bugg_ID, deployment_start, deployment_end)]

        #data = [DATASET_PATH, country_folder, bugg_folder, conf_folder, file, deploymentID, country, cluster_name, site_name, float(lat), float(long)]

        data = [ format_metadata( f, deploymentID, country, cluster_name, site_name, lat, long ) for f in filtered_missed_files]

        print(deploymentID, cluster_name, site_name, len(data), "missing files")
        files_data.extend(data)


    # === SPLIT FILES INTO CHUNKS ===
    max_chunk_size = 30000
    N_JOBS =  math.ceil( len(files_data) / max_chunk_size)
    chunk_size = math.ceil(len(files_data) / N_JOBS)

    #if os.path.exists(OUTPUT_CHUNK_FOLDER):
    #    shutil.rmtree(OUTPUT_CHUNK_FOLDER)
    os.makedirs(OUTPUT_CHUNK_FOLDER, exist_ok=False) 

    for i in range(N_JOBS):
        chunk_files = files_data[i * chunk_size: (i + 1) * chunk_size]
        # Write the chunk file, including the additional arguments for each file

        with open(os.path.join(PIPE_LINE_PATH, OUTPUT_CHUNK_FOLDER, f"file_chunks_{i}.txt"), "w") as f:
            for file in chunk_files:
                # Write file path along with the additional arguments
                f.write(f"{file}\n")


    print(f"Split {MONTH_SELECTION} : {len(files_data)} files into {N_JOBS} chunks.")

    print(f" Inference will take approximately {chunk_size*8/60/60:.1f} hours per job" )


    # === CREATE SBATCH FILE ===
    SBATCH_TEMPLATE = f"""#!/bin/bash
    #SBATCH --job-name=tabmon_pipeline
    #SBATCH --partition=besteffort         
    #SBATCH --output=slurm_output_files_missed/slurm_output_%A_%a.out
    #SBATCH --array=0-{N_JOBS-1}
    #SBATCH --gres=gpu:1  # Request 1 GPU per job
    #SBATCH --cpus-per-task=2  
    #SBATCH --nodes=1                
    #SBATCH --mem-per-cpu=16G        
    #SBATCH --time=7-00:00:00    
    

    echo "Executing on the machine:" $(hostname)
    echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
    echo "Number of tasks: $SLURM_NTASKS"
    echo "Number of CPUs per task: $SLURM_CPUS_PER_TASK"
    echo "Processing chunk $SLURM_ARRAY_TASK_ID"

    # Pass additional parameters
    python {PYTHON_SCRIPT} {OUTPUT_CHUNK_FOLDER}/file_chunks_$SLURM_ARRAY_TASK_ID.txt 
    """

    with open(os.path.join(PIPE_LINE_PATH, SBATCH_OUTPUT_FILE), "w") as f:
        f.write(SBATCH_TEMPLATE)

    print(f"SBATCH script '{SBATCH_OUTPUT_FILE}' created.")






"""
files_data = []

for month_index, month in enumerate(MONTH_SELECTION) :

    month_analyzed_count = 0
    month_missed_count = 0

    chunk_folder = f"chunk_files_{month}"

    chunk_file_list = [f for f in os.listdir(chunk_folder) if f.endswith(".txt")]

    already_analyzed_recordings = []

    for chunk_file in chunk_file_list :
        with open(os.path.join(chunk_folder, chunk_file), encoding='utf-8') as file:
            chunk_lines = file.readlines()
            
        for line in chunk_lines: 
            parts = ast.literal_eval(line)
            recording_path = os.path.join(parts[0], parts[1],parts[2],parts[3], parts[4] )
            already_analyzed_recordings.append(recording_path)
            
    print("Month", month , " : ", len(already_analyzed_recordings) , "analyzed recordings in total in chunk files" )



    for index, meta_data_row in META_DATA_DF.iterrows():

        missed_count = 0
        analyzed_count = 0

        deploymentID = meta_data_row["DeploymentID"]
        bugg_ID = meta_data_row["DeviceID"]
        country = meta_data_row["Country"]
        cluster_name = meta_data_row["Cluster"]
        site_name = meta_data_row["Site"]
        lat = meta_data_row["Latitude"]
        long = meta_data_row["Longitude"]
        start_date = meta_data_row["deploymentBeginDate"]
        start_time = meta_data_row["deploymentBeginTime"]
        end_date = meta_data_row["deploymentEndDate"]
        end_time = meta_data_row["deploymentEndTime"]

        deployment_start = datetime.strptime(f"{start_date} {start_time}", "%d/%m/%Y %H:%M:%S")

        if end_date == "" and end_time == "" :
            deployment_end = datetime(3000, 1, 1)
        else :
            deployment_end = datetime.strptime(f"{end_date} {end_time}", "%d/%m/%Y %H:%M:%S")

        
        country_folder = COUNTRY_TO_FOLDER[country]

        bugg_folder_list = [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder)) if os.path.isdir(os.path.join(DATASET_PATH, country_folder, f))]

        bugg_folder = [s for s in bugg_folder_list if s.endswith(bugg_ID)]

        if len(bugg_folder) == 1 :

            bugg_folder = bugg_folder[0]
            conf_folder_list =  [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder, bugg_folder)) if os.path.isdir(os.path.join(DATASET_PATH, country_folder, bugg_folder, f))]

            for conf_folder in conf_folder_list:

                
                recording_files = [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder, bugg_folder, conf_folder)) if f.endswith(".mp3")]


                for file in recording_files:
                        
                    file_year_month = get_file_year_month(file)

                    if file_year_month in month:
                        
                        try : 
                            file_date = get_file_date(file)
                            
                            if file_date > deployment_start and file_date < deployment_end :

                                recording_path = os.path.join(DATASET_PATH, country_folder, bugg_folder, conf_folder, file)

                                ##evaluate if the recording have already been analyzed:
                                if recording_path not in already_analyzed_recordings:
                                    data = [DATASET_PATH, country_folder, bugg_folder, conf_folder, file, deploymentID, country, cluster_name, site_name, float(lat), float(long)]
                                    files_data.append(data)
                                    missed_count = missed_count+1
                                    month_missed_count = month_missed_count+1


                                else:
                                    analyzed_count = analyzed_count+1
                                    month_analyzed_count = month_analyzed_count+1
                                
                        except Exception as e: 
                            print("Unable to get date from ", file)
                            print(e)


        else :
            print("No bugg folder for", deploymentID)


        if missed_count > 0:
            print(f"{analyzed_count} analyzed files and {missed_count} missed files in {month} for", deploymentID, deployment_start, deployment_end)


    print(f"{month_analyzed_count} analyzed files and {month_missed_count} missed files in {month}" )

"""






