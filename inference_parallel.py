import sys
import json
import os
import time
import ast
import psutil

RESULT_FILES_FOLDER = "result_files"

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
                    parts = ast.literal_eval(line)
                    
                    # Extract the parts as required
                    dataset_path = parts[0]
                    bugg = parts[1]
                    conf = parts[2]
                    file = parts[3]
                    country = parts[4]
                    site = parts[5]
                    lat = parts[6]
                    long = parts[7]

                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")

            # Process the file with the passed parameters
            # something like analyze_file(dataset_path, bugg, conf, file, country, lat, long, job_id)


            if i % 100 == 0:
                print_time_information(time_start, i, number_of_files)


    print("End processing")
    print_time_information(time_start, i, number_of_files)
