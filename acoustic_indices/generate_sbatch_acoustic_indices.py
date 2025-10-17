import os
import math
import pandas as pd
import numpy as np
import shutil
import random
from datetime import datetime, timedelta
random.seed(11)
today_date = datetime.today().strftime('%Y-%m-%d')


# === CONFIGURATION ===
DATASET_PATH = "/DYNI/tabmon/tabmon_data/proj_tabmon_NINA_NL"

#DURATION = 60 # sec

STEP = timedelta(minutes=15)
START_DATE = datetime(2025, 8, 31)
END_DATE = datetime(2025, 9, 22)


COUNTRY_TO_FOLDER = {
    "Norway": "proj_tabmon_NINA",
    "Spain": "proj_tabmon_NINA_ES",
    "France": "proj_tabmon_NINA_FR",
    "Netherlands": "proj_tabmon_NINA_NL"
}


PIPE_LINE_PATH = "./"
SBATCH_OUTPUT_FILE = "run_acoustic_indices.sh"
PYTHON_SCRIPT = "compute_acoustic_indices.py" 
MONTH_PRINT = f"{START_DATE.date()}_to_{END_DATE.date()}"

CHUNK_FILES_FOLDER = f"chunk_files_{MONTH_PRINT}"  

if os.path.exists(CHUNK_FILES_FOLDER):
    shutil.rmtree(CHUNK_FILES_FOLDER)
os.makedirs(CHUNK_FILES_FOLDER, exist_ok=False) 

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


chunk_sizes = []
N_JOBS = 0

if __name__ == "__main__":


    
    bugg_folder_list = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

    for bugg_folder in bugg_folder_list:

        #df_indices_list = []
        #analyzed_file_list = []
        #datetime_list = []

        conf_folder_list =  [f for f in os.listdir(os.path.join(DATASET_PATH, bugg_folder)) if os.path.isdir(os.path.join(DATASET_PATH, bugg_folder, f))]
        
        files_selected = []
        chunk_files = []

        for conf_folder in conf_folder_list:

            recording_files = [f for f in os.listdir(os.path.join(DATASET_PATH, bugg_folder, conf_folder)) if f.endswith(".mp3")]

            files_and_dates = [(f, get_file_date(f))  for f in recording_files ]
            files_and_dates.sort(key=lambda x: x[1])
    
            time_ref = files_and_dates[0][1]

            #files_and_dates = files_and_dates[20000:20100]

            for file, file_time in files_and_dates:

                if file_time > START_DATE and file_time < END_DATE : # if in the time period
            
                    if file_time >= time_ref:   # if more than 15 min since last file
                        files_selected.append((file, file_time))
                        time_ref = file_time + STEP
                        chunk_files.append( [DATASET_PATH, bugg_folder, conf_folder, file, MONTH_PRINT] )


        if len(chunk_files) > 0: 

            print(len(chunk_files), "files for", bugg_folder)

            with open(os.path.join(PIPE_LINE_PATH, CHUNK_FILES_FOLDER, f"file_chunks_{N_JOBS}.txt"), "w") as f:
                for file in chunk_files:
                    # Write file path along with the additional arguments
                    f.write(f"{file}\n")
                    
            N_JOBS = N_JOBS +1 


print(f"Split files into {N_JOBS} chunks (one per bugg).")


# === CREATE SBATCH FILE ===
SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH --job-name=acoustic_indices
#SBATCH --partition=cpu         
#SBATCH --output=slurm_output_files_{MONTH_PRINT}/slurm_output_%A_%a.out
#SBATCH --array=0-{N_JOBS-1}             
#SBATCH --mem-per-cpu=32G        
#SBATCH --time=7-00:00:00    
  

echo "Executing on the machine:" $(hostname)
echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Processing chunk $SLURM_ARRAY_TASK_ID"

# Pass additional parameters
python {PYTHON_SCRIPT} {CHUNK_FILES_FOLDER}/file_chunks_$SLURM_ARRAY_TASK_ID.txt 
"""

with open(os.path.join(PIPE_LINE_PATH, SBATCH_OUTPUT_FILE), "w") as f:
    f.write(SBATCH_TEMPLATE)

print(f"SBATCH script '{SBATCH_OUTPUT_FILE}' created.")