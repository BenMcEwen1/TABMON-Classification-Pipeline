import os
import math
import pandas as pd
import numpy as np
import shutil
import random

# === CONFIGURATION ===
N_JOBS = 7  # Number of parallel jobs

SUBSAMPLE_FACTOR = 100 # select randomly only 1/SUBSAMPLE_FACTOR of the file (for testing)
random.seed(10)

DATASET_PATH = "/DYNI/tabmon/tabmon/proj_tabmon_NINA" 
SBATCH_OUTPUT_FILE = "parallel_inference.sbatch"
PYTHON_SCRIPT = "inference_parallel.py" 
CHUNK_FILES_FOLDER = "chunk_files"
RESULT_FILES_FOLDER = "result_files"

META_DATA_PATH = "Bugg deployment form.csv"
META_DATA_DF = pd.read_csv(os.path.join(DATASET_PATH, META_DATA_PATH) , encoding='utf-8')


def get_device_ID(bugg_folder_name):
    """Get device ID (last digits afters 0000000) from the bugg folder name."""
    indices = [i for i, char in enumerate(bugg_folder_name) if char == '0']
    indices = np.array(indices)
    change_indices = np.append(np.diff(indices)-1, 1)
    last_zero_index = np.where(change_indices != 0)[0][0]
    return bugg_folder_name[indices[last_zero_index]+1:]


## get the list of bugg_folders
bugg_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

files_data = []
## loop over bugg folders
for bugg in bugg_folders:

    #Get metadata 
    bugg_ID = get_device_ID(bugg)
    meta_data_row = META_DATA_DF[META_DATA_DF["8. DeviceID: last digits of the serial number (ex: RPiID-100000007ft35sm --> 7ft35sm)"] == bugg_ID]
    country = meta_data_row["1. Country"].iloc[0]
    site_name = "site_name" # site names are going to be added to the metadata 
    lat = meta_data_row["4. Latitude: decimal degree, WGS84 (ex: 64.65746)"].iloc[0]
    long = meta_data_row["5. Longitude: decimal degree, WGS84 (ex: 5.37463)"].iloc[0]

    ## get the list of conf folders
    conf_folders =  [f for f in os.listdir(os.path.join(DATASET_PATH, bugg)) if os.path.isdir(os.path.join(DATASET_PATH, bugg, f))]
    
    ## loop over conf folders (if in the future there is more than one conf file per bugg)
    for conf in conf_folders:
        recording_files = [f for f in os.listdir(os.path.join(DATASET_PATH, bugg, conf)) if f.endswith(".mp3")]

        #Subsample the list, for testing
        sample_size = int(len(recording_files)/SUBSAMPLE_FACTOR)
        recording_files = random.sample(recording_files, sample_size)

        for file in recording_files:
            data = [DATASET_PATH, bugg, conf, file, country, site_name, float(lat), float(long) ]
            files_data.append(data)



# === SPLIT FILES INTO CHUNKS ===
chunk_size = math.ceil(len(files_data) / N_JOBS)

if os.path.exists(CHUNK_FILES_FOLDER):
    shutil.rmtree(CHUNK_FILES_FOLDER)
os.makedirs(CHUNK_FILES_FOLDER, exist_ok=False) 

# Write the aggregated results for this job to a file
if os.path.exists(RESULT_FILES_FOLDER):
    shutil.rmtree(RESULT_FILES_FOLDER)


for i in range(N_JOBS):
    chunk_files = files_data[i * chunk_size: (i + 1) * chunk_size]
    # Write the chunk file, including the additional arguments for each file
    
    with open(os.path.join(CHUNK_FILES_FOLDER, f"file_chunks_{i}.txt"), "w") as f:
        for file in chunk_files:
            # Write file path along with the additional arguments
            f.write(f"{file}\n")


print(f"Split {len(files_data)} files into {N_JOBS} chunks.")


# === CREATE SBATCH FILE ===
SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH --job-name=tabmon_pipeline
#SBATCH --partition=gpu         
#SBATCH --output=slurm_output_files/slurm_output_%A_%a.out
#SBATCH --array=0-{N_JOBS-1}
#SBATCH --gres=gpu:1  # Request 1 GPU per job
#SBATCH --cpus-per-task=2  
#SBATCH --nodes=1                
#SBATCH --mem-per-cpu=4G        
#SBATCH --time=0-24:00:00    
  

echo "Executing on the machine:" $(hostname)
echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Processing chunk $SLURM_ARRAY_TASK_ID"

# Pass additional parameters
python {PYTHON_SCRIPT} {CHUNK_FILES_FOLDER}/file_chunks_$SLURM_ARRAY_TASK_ID.txt 
"""

with open(SBATCH_OUTPUT_FILE, "w") as f:
    f.write(SBATCH_TEMPLATE)

print(f"SBATCH script '{SBATCH_OUTPUT_FILE}' created.")
