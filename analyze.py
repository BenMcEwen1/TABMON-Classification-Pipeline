#!/usr/bin/env python3

import time
from AvesEcho.models import run_algorithm

start_time = time.time()

if __name__ == "__main__":
    result_file = 'AvesEcho/outputs/analysis-results.json'

    run_algorithm(avesecho_mapping='AvesEcho/inputs/list_AvesEcho.csv', result_file=result_file)

    # Compute the elapsed time in seconds
    elapsed_time = time.time() - start_time
    print(elapsed_time)