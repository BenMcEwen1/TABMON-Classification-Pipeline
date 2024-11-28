#!/bin/bash

# The behavior of this script depends on the algorithm_mode and input_directory command-line options.
algorithm_mode=""
input_directory=""

# Parse the relevant command-line options (algorithm_mode and input_directory).
for arg in "$@"; do
    case $arg in
        --algorithm_mode=*)
            algorithm_mode="${arg#*=}"
            ;;
        --i=*)
            input_directory="${arg#*=}"
            ;;
    esac
done

# Only run this when algorithm_mode is "directories".
if [ "$algorithm_mode" == "directories" ]; then
    # This line is needed for running AvesEcho-v1 on Snellius. (When converting the Docker container to Apptainer, the
    # working directory WORKDIR is not handled correctly. This is fixed by explicitly changing the directory here.)
    cd /app || exit

    # These two environment variables have to be set on Snellius.
    export NUMBA_CACHE_DIR=/app/tmp
    export MPLCONFIGDIR=/app/tmp
fi

python3 analyze.py "$@";

# Only run this when algorithm_mode is "directories".
if [ "$algorithm_mode" == "directories" ]; then
    # After analyzing the current batch, remove the audio files from the input directory to prepare for the next batch.
    rm "$input_directory"/*.*
fi
