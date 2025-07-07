## generate sbatch and chunk files for jobs that have been interrupted

import os
import math
import pandas as pd
import numpy as np
import shutil
import random
from datetime import datetime
random.seed(11)
today_date = datetime.today().strftime('%Y-%m-%d')


MONTH_PRINT = "2025-05"
PIPE_LINE_PATH = "./"
SBATCH_OUTPUT_FILE = "resume_jobs.sh"
PYTHON_SCRIPT = "inference_parallel.py" 

INPUT_CHUNK_FOLDER = f"chunk_files_{MONTH_PRINT}"
INPUT_SLURM_FOLDER = "slurm_output_files"

OUTPUT_CHUNK_FOLDER = f"chunk_files_resume_{MONTH_PRINT}_{today_date}"

os.makedirs(OUTPUT_CHUNK_FOLDER, exist_ok=False) 


slurm_file_list = [f for f in os.listdir(os.path.join(PIPE_LINE_PATH,INPUT_SLURM_FOLDER)) if f.endswith(".out")]
slurm_file_list = sorted(slurm_file_list)


for index, slurm_file in enumerate(slurm_file_list):

    chunk_id = (slurm_file.split(".")[0]).split("_")[-1]
    print(chunk_id)
    
    with open(os.path.join(PIPE_LINE_PATH, INPUT_SLURM_FOLDER, slurm_file), encoding='utf-8') as file:
        slurm_lines = file.readlines()
        
    if slurm_lines[-2] == "End processing\n":

        print("Job completed")

    else:
        
        stop_file_id = int((len(slurm_lines)-7) / 3.1)-3
        stop_file_id = max([stop_file_id,0])

        print("Job interupted at file n°" , stop_file_id)

        chunk_file = f"file_chunks_{chunk_id}.txt"
        with open(os.path.join(PIPE_LINE_PATH, INPUT_CHUNK_FOLDER, chunk_file), encoding='utf-8') as file:
            chunk_lines = file.readlines()

        chunk_lines_left = chunk_lines[stop_file_id:]

        with open(os.path.join(PIPE_LINE_PATH, OUTPUT_CHUNK_FOLDER, chunk_file), "w") as f:
            for data in chunk_lines_left:
                f.write(data)


N_JOBS = len(slurm_file_list) 
print(f"Resume {MONTH_PRINT} : {N_JOBS} chunks" )

# === CREATE SBATCH FILE ===
SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH --job-name=tabmon_pipeline
#SBATCH --partition=all         
#SBATCH --output=slurm_output_files_resume_{MONTH_PRINT}/slurm_output_%A_%a.out
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