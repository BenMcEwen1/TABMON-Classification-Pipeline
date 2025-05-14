import os
import math
import pandas as pd
import numpy as np
import shutil
import random
from datetime import datetime

today_date = datetime.today().strftime('%Y-%m-%d')

# === CONFIGURATION ===
N_JOBS = 1  # Number of parallel jobs

#Already processed : "2024-03","2024-04"
MONTH_SELECTION = ["2024-11", "2024-10", "2024-09", "2024-08", "2024-07", "2024-06", "2024-05", "2025-05", "2025-04", "2025-03", "2025-02", "2025-01"]


# useless [bugg ID - conf_name]  deployed in 2024 with the mic problem
USELESS_BUGGS = [["49662376", "conf_20240314_TABMON"], ["23646e76", "conf_20240314_TABMON"], ["ed9fc668", "conf_20240314_TABMON"], ["add20a52", "conf_20240314_TABMON"], ["3a6c5dee", "conf_20240314_TABMON"]] 


SUBSAMPLE_FACTOR = 1 # select randomly only 1/SUBSAMPLE_FACTOR of the file (for testing)
random.seed(11)

#DATASET_PATH = "audio/test_bugg/" 
DATASET_PATH = "./audio/tabmon_subset/" 
COUNTRY_FOLDERS_LIST = ["proj_tabmon_NINA", "proj_tabmon_NINA_ES", "proj_tabmon_NINA_FR", "proj_tabmon_NINA_NL"]
PIPE_LINE_PATH = "./"
SBATCH_OUTPUT_FILE = "parallel_inference.sh"
PYTHON_SCRIPT = "inference_parallel.py" 
CHUNK_FILES_FOLDER = f"chunk_files_{today_date}"
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


def get_file_date(bugg_file_name):
    """Get file year-month"""
    fulldate = bugg_file_name.split("-")
    year_month = fulldate[0] + '-' + fulldate[1]
    return year_month


missing_dates = []
files_data = []
## loop over country folders (ex: proj_tabmon_NINA_ES")
for country_folder in COUNTRY_FOLDERS_LIST:
    print(country_folder)

    ## get the list of bugg_folders
    bugg_folder_list = [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder)) if os.path.isdir(os.path.join(DATASET_PATH, country_folder, f))]

    ## loop over bugg folders
    for bugg_folder in bugg_folder_list:

        print(country_folder,bugg_folder)
        #Get metadata 
        bugg_ID = get_device_ID(bugg_folder)

        if bugg_ID in META_DATA_DF["8. DeviceID: last digits of the serial number (ex: RPiID-100000007ft35sm --> 7ft35sm)"].tolist() :
            meta_data_row = META_DATA_DF[META_DATA_DF["8. DeviceID: last digits of the serial number (ex: RPiID-100000007ft35sm --> 7ft35sm)"] == bugg_ID]
            country = meta_data_row["1. Country"].iloc[0]
            site_name = "site_name" # site names are going to be added to the metadata 
            lat = meta_data_row["4. Latitude: decimal degree, WGS84 (ex: 64.65746)"].iloc[0]
            long = meta_data_row["5. Longitude: decimal degree, WGS84 (ex: 5.37463)"].iloc[0]

            ## get the list of conf folders
            conf_folder_list =  [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder, bugg_folder)) if os.path.isdir(os.path.join(DATASET_PATH, country_folder, bugg_folder, f))]
            
            ## loop over conf folders (if in the future there is more than one conf file per bugg)
            for conf_folder in conf_folder_list:

                if [bugg_ID, conf_folder] not in USELESS_BUGGS:

                    recording_files = [f for f in os.listdir(os.path.join(DATASET_PATH, country_folder, bugg_folder, conf_folder)) if f.endswith(".mp3")]

                    #Subsample the list, for testing
                    sample_size = int(len(recording_files)/SUBSAMPLE_FACTOR)
                    recording_files = random.sample(recording_files, sample_size)

                    for file in recording_files:
                        
                        file_year_month = get_file_date(file)

                        if file_year_month in MONTH_SELECTION:

                            data = [DATASET_PATH, country_folder, bugg_folder, conf_folder, file, country, site_name, float(lat), float(long) ]
                            files_data.append(data)
                        else:
                            missing_dates.append(file_year_month)
                            print([bugg_ID, conf_folder], " not in the month selection")
                else :
                    print([bugg_ID, conf_folder], " in list of not usable buggs")
        else:
            print("No metadata for ", country_folder, bugg_folder)

print(f"Total number of files: {len(files_data)}")
print(f"Missing dates: {missing_dates}")





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
    
    with open(os.path.join(PIPE_LINE_PATH, CHUNK_FILES_FOLDER, f"file_chunks_{i}.txt"), "w") as f:
        for file in chunk_files:
            # Write file path along with the additional arguments
            f.write(f"{file}\n")


print(f"Split {MONTH_SELECTION} : {len(files_data)} files into {N_JOBS} chunks.")


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