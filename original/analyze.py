# -*- coding: utf-8 -*-
# @Time    : 19/03/24 14:30
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl
# @File    : analyze.py


import argparse
import os
import time

start_time = time.time()

from algorithm_mode import AlgorithmMode
from models import run_algorithm
from util import get_base_name


default_algorithm_mode = os.getenv("ALGORITHM_MODE", AlgorithmMode.DIRECTORIES.value)

# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--slist', type=str, default='inputs/list_sp_ml.csv', help='Path to the species list.')
parser.add_argument('--flist', type=str, default='inputs/species_list_nl.csv', help='Path to the filter list of species.')
parser.add_argument('--i', type=str, default='audio/NH-11_20240415_062840.WAV', help='Input audio sample.')
parser.add_argument('--o', type=str, default='tmp/avesecho', help='Output directory for temporary audio chunks.')
parser.add_argument('--mconf', type=float, default=None, help='Minimum confidence threshold for predictions.')
parser.add_argument('--lat', type=float, default=None, help='Latitude for geographic filtering.')
parser.add_argument('--lon', type=float, default=None, help='Longitude for geographic filtering.')
parser.add_argument('--add_filtering', action='store_true', help='Enable geographic filtering.')
parser.add_argument('--add_csv', action='store_true', help='Save predictions to a CSV file.')
parser.add_argument('--maxpool', action='store_true', help='Use model for generating temporally-summarised output.')
parser.add_argument('--model_name', type=str, default='fc', help='Name of the model to use.')
parser.add_argument("--algorithm_mode", default=default_algorithm_mode, help="Use input/output directories or an endpoint.")

if __name__ == "__main__":

    args = parser.parse_args()

    if os.path.isfile(args.i):
        # Results are saved to a file containing the name of the input file
        result_file = get_base_name(args.i)
    else:
        result_file = 'outputs/analysis-results.json'

    run_algorithm(args, avesecho_mapping='inputs/list_AvesEcho.csv', result_file=result_file)

    # Compute the elapsed time in seconds
    elapsed_time = time.time() - start_time

    # Print the time it took
    description = f"audio file {args.i}" if os.path.isfile(args.i) else "batch of audio files"
    print(f"It took {elapsed_time:.2f}s to analyze the {description}.")
