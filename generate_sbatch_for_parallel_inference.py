import os
import math
import pandas as pd
import numpy as np
import shutil
import random
from datetime import datetime
random.seed(11)
today_date = datetime.today().strftime('%Y-%m-%d')


# === CONFIGURATION ===
N_JOBS = 20  # Number of parallel jobs

SUBSAMPLE_FACTOR = 1 # select randomly only 1/SUBSAMPLE_FACTOR of the file (for testing)

#Already processed : "2025-01", "2025-02", "2025-03"
MONTH_SELECTION = ["2025-04"]

# useless [bugg ID - conf_name]  deployed in 2024 with the mic problem
USELESS_BUGGS = [["49662376", "conf_20240314_TABMON"], ["23646e76", "conf_20240314_TABMON"], ["ed9fc668", "conf_20240314_TABMON"], ["add20a52", "conf_20240314_TABMON"], ["3a6c5dee", "conf_20240314_TABMON"]] 


DATASET_PATH = "/DYNI/tabmon/tabmon_data" 

COUNTRY_TO_FOLDER = {
    "Norway": "proj_tabmon_NINA",
    "Spain": "proj_tabmon_NINA_ES",
    "France": "proj_tabmon_NINA_FR",
    "Netherlands": "proj_tabmon_NINA_NL"
}


PIPE_LINE_PATH = "./"
SBATCH_OUTPUT_FILE = "parallel_inference.sh"
PYTHON_SCRIPT = "inference_parallel.py" 
MONTH_PRINT = ";".join(str(x) for x in MONTH_SELECTION)
CHUNK_FILES_FOLDER = f"chunk_files_{MONTH_PRINT}"


META_DATA_PATH = "site_info.csv"

META_DATA_DF = pd.read_csv(os.path.join(DATASET_PATH, META_DATA_PATH) , encoding='utf-8')



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
for index, meta_data_row in META_DATA_DF.iterrows():


    deployementID = meta_data_row["9. DeploymentID: countryCode_deploymentNumber_DeviceID (e.g. NO_1_ 7ft35sm)"]

    print(deployementID)

    bugg_ID = meta_data_row["8. DeviceID: last digits of the serial number (ex: RPiID-100000007ft35sm --> 7ft35sm)"]
    country = meta_data_row["1. Country"]
    site_name = meta_data_row["Site"]
    lat = meta_data_row["4. Latitude: decimal degree, WGS84 (ex: 64.65746)"]
    long = meta_data_row["5. Longitude: decimal degree, WGS84 (ex: 5.37463)"]
    
    d_start_date = meta_data_row["2. Date"]
    d_start_hour = meta_data_row["3. Time (UTC!!! Check here  https://www.utctime.net/)"]

    deployement_start = datetime.strptime(f"{d_start_date} {d_start_hour}", "%d/%m/%Y %H:%M:%S")
    deployement_end  = datetime(2030, 1, 1)

    #print(deployementID, bugg_ID, country, site_name, lat, long   , deployement_start, deployement_end)

    country_folder = COUNTRY_TO_FOLDER[country]

    bugg_folder_list = [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder)) if os.path.isdir(os.path.join(DATASET_PATH, country_folder, f))]

    bugg_folder = [s for s in bugg_folder_list if s.endswith(bugg_ID)]

    if len(bugg_folder) == 1 :

        bugg_folder = bugg_folder[0]
        conf_folder_list =  [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder, bugg_folder)) if os.path.isdir(os.path.join(DATASET_PATH, country_folder, bugg_folder, f))]

        for conf_folder in conf_folder_list:

            
            recording_files = [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder, bugg_folder, conf_folder)) if f.endswith(".mp3")]

            #Subsample the list, for testing
            sample_size = int(len(recording_files)/SUBSAMPLE_FACTOR)
            recording_files = random.sample(recording_files, sample_size)

            for file in recording_files:
                    
                file_year_month = get_file_year_month(file)

                if file_year_month in MONTH_SELECTION:
                    
                    file_date = get_file_date(file)
                    
                    print(deployementID, deployement_start, file_date,deployement_end )

                    if file_date > deployement_start and file_date < deployement_end :
                        data = [DATASET_PATH, country_folder, bugg_folder, conf_folder, file, country, site_name, float(lat), float(long), deployementID ]
                        files_data.append(data)
                        
                        print(data)
    
    else :
        print("No bugg folder for", deployementID)


print(f"Total number of files: {len(files_data)}")



# === SPLIT FILES INTO CHUNKS ===
chunk_size = math.ceil(len(files_data) / N_JOBS)

if os.path.exists(CHUNK_FILES_FOLDER):
    shutil.rmtree(CHUNK_FILES_FOLDER)
os.makedirs(CHUNK_FILES_FOLDER, exist_ok=False) 



for i in range(N_JOBS):
    chunk_files = files_data[i * chunk_size: (i + 1) * chunk_size]
    # Write the chunk file, including the additional arguments for each file
    
    with open(os.path.join(PIPE_LINE_PATH, CHUNK_FILES_FOLDER, f"file_chunks_{i}.txt"), "w") as f:
        for file in chunk_files:
            # Write file path along with the additional arguments
            f.write(f"{file}\n")


print(f"Split {MONTH_SELECTION} : {len(files_data)} files into {N_JOBS} chunks.")
print(f" Inference will take approximately {chunk_size*30/60/60:.1f} hours per job" )

# === CREATE SBATCH FILE ===
SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH --job-name=tabmon_pipeline
#SBATCH --partition=all         
#SBATCH --output=slurm_output_files/slurm_output_%A_%a.out
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
python {PYTHON_SCRIPT} {CHUNK_FILES_FOLDER}/file_chunks_$SLURM_ARRAY_TASK_ID.txt 
"""

with open(os.path.join(PIPE_LINE_PATH, SBATCH_OUTPUT_FILE), "w") as f:
    f.write(SBATCH_TEMPLATE)

print(f"SBATCH script '{SBATCH_OUTPUT_FILE}' created.")