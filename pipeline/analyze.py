import argparse
import os
import time

from pipeline.algorithm_mode import AlgorithmMode
from pipeline.models import run_algorithm
from app.schema import PipelineSchema
from app.services import normalise
from app.database import SessionLocal


start_time = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))

default_algorithm_mode = os.getenv("ALGORITHM_MODE", AlgorithmMode.DIRECTORIES.value)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slist', type=str, default=f'{current_dir}/inputs/list_sp_ml.csv', help='Path to the species list.')
parser.add_argument('--i', type=str, default=f'{current_dir}/audio/NH-11_20240415_062840.WAV', help='Input audio sample.')
parser.add_argument('--flist', type=str, default=None, help='Path to the filter list of species.')
parser.add_argument('--device_id', type=str, default=None, required=True, help='Device id - last digits of the serial number (i.e. RPiID-100000007ft35sm --> 7ft35sm).')
parser.add_argument('--country', type=float, default=None, help='Country')
parser.add_argument('--lat', type=float, default=None, help='Latitude for geographic filtering.') # TODO: If lat/lng specified location filter applied
parser.add_argument('--lng', type=float, default=None, help='Longitude for geographic filtering.')
parser.add_argument('--model_name', type=str, default='birdnet', help='Name of the model to use.')
parser.add_argument('--model_checkpoint', type=str, default=None, help='Model checkpoint - base if not specified.')

def run(args, db=None):
    args = PipelineSchema(**vars(args)) # Additional validation
    predictions = run_algorithm(args)
    if db:
        status = normalise(predictions, db)
    else:
        db = SessionLocal()
        status = normalise(predictions, db)
        db.close()
    return status

if __name__ == "__main__":
    run(parser.parse_args())
    print(f"It took {(time.time() - start_time):.2f}s to analyze the audio files.")