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


MONTH_SELECTION = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"]

DATASET_PATH = "/DYNI/tabmon/tabmon_data" 
PIPE_LINE_PATH = "./"
SBATCH_OUTPUT_FILE = "inference_missed_files.sh"
PYTHON_SCRIPT = "inference_parallel.py" 

OUTPUT_CHUNK_FOLDER = "chunk_files_missed"


META_DATA_PATH = "site_info.csv"
META_DATA_DF = pd.read_csv(os.path.join(DATASET_PATH, META_DATA_PATH) , encoding='utf-8')
META_DATA_DF = META_DATA_DF.fillna("")


COUNTRY_TO_FOLDER = {
    "Norway": "proj_tabmon_NINA",
    "Spain": "proj_tabmon_NINA_ES",
    "France": "proj_tabmon_NINA_FR",
    "Netherlands": "proj_tabmon_NINA_NL"
}


def get_file_year_month(bugg_file_name):
    """Get file year-month"""
    fulldate = bugg_file_name.split("-")
    year_month = fulldate[0] + '-' + fulldate[1]
    return year_month

def get_file_date(bugg_file_name):
    date = bugg_file_name.replace('.mp3', '')
    date = date.replace('_', ':')
    date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ")
    return date


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



# === SPLIT FILES INTO CHUNKS ===

max_chunk_size = 9000
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

print(f" Inference will take approximately {chunk_size*23/60/60:.1f} hours per job" )


# === CREATE SBATCH FILE ===
SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH --job-name=tabmon_pipeline
#SBATCH --partition=GPU         
#SBATCH --output=slurm_output_files_missed/slurm_output_%A_%a.out
#SBATCH --array=0-{N_JOBS-1}
#SBATCH --gres=gpu:1  # Request 1 GPU per job
#SBATCH --cpus-per-task=2  
#SBATCH --nodes=1                
#SBATCH --mem-per-cpu=4G        
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