import sys
import json
import os
import time
import ast
import psutil
from pipeline.analyze import run
from types import SimpleNamespace
import numpy as np
import traceback

RESULT_FILES_FOLDER = "result_files"

def get_device_ID(bugg_folder_name):
    """Get device ID (last digits afters 0000000) from the bugg folder name."""
    indices = [i for i, char in enumerate(bugg_folder_name) if char == '0']
    indices = np.array(indices)
    change_indices = np.append(np.diff(indices)-1, 1)
    last_zero_index = np.where(change_indices != 0)[0][0]
    return bugg_folder_name[indices[last_zero_index]+1:]

# Function to get current memory usage of the Python process
def get_memory_usage():
    process = psutil.Process(os.getpid())  # Get the current process ID
    memory_info = process.memory_info()  # Get memory usage information
    return memory_info.rss  # Returns memory in bytes (Resident Set Size)


def print_time_information(time_start, i, number_of_files):
    elapsed_time = (time.time() - time_start)
    sec_per_file = elapsed_time/(i+1)
    number_of_remaining_files = number_of_files - i
    remaining_time = (number_of_remaining_files * sec_per_file)/60

    print(f"Processed {i+1}/{number_of_files} files in {elapsed_time/60:.1f} min, {sec_per_file:.1f} sec per file, {remaining_time:.1f} min remaining, Memory used: {get_memory_usage() / (1024 * 1024):.0f} MB", flush=True )

if __name__ == "__main__":
    time_start = time.time()
    i = 0

    chunk_file = sys.argv[1] 
    job_id = chunk_file.split('_')[-1].split('.')[0] #job identifier based on the chunk name

    print(f"Start processsing {chunk_file}")

    results_files = []

    with open(chunk_file, "r") as f:

        number_of_files = sum(1 for line in f)
        f.seek(0) # Reset file pointer to the beginning of the file before reading the lines
        print(f"Start processing {number_of_files} files", flush=True)

        for i, line in enumerate(f):
            # Remove leading/trailing whitespace and brackets, then use ast.literal_eval() to safely parse the list
            line = line.strip()  # Remove leading/trailing whitespace

            if line:  # Check if the line is not empty
                try:
                    # Safely parse the list (it will turn the string into an actual Python list)
                    
                    print(i)
                    
                    parts = ast.literal_eval(line)
                    
                    # Extract the parts as required
                    dataset_path = parts[0]
                    country_folder = parts[1]
                    bugg = parts[2]
                    conf = parts[3]
                    file = parts[4]
                    deploymentID = parts[5]
                    country = parts[6]
                    cluster = parts[7]
                    site = parts[8]
                    lat = parts[9]
                    long = parts[10]
                    


                    args = {
                            "slist": 'pipeline/inputs/list_sp_ml.csv',
                            "flist": None,
                            "i": os.path.join(dataset_path, country_folder, bugg, conf, file),
                            "device_id": get_device_ID(bugg),
                            "country": country,
                            "lat": lat,
                            "lng": long,
                            "model_name": 'birdnet',
                            "model_checkpoint": None,
                            "date_updated": None,
                            "date_deployed": None
                    }

                    args = SimpleNamespace(**args)
                    run(args, id=job_id)

                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    traceback.print_exc()

            if (number_of_files != 0) and (i % 10 == 0):
                print_time_information(time_start, i, number_of_files)

    print("End processing")
    print_time_information(time_start, i, number_of_files)
