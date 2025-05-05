import time
import os
import argparse
import pandas as pd

from pipeline.models import run_algorithm
from app.schema import PipelineSchema

start_time = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = f"{current_dir}/outputs"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slist', type=str, default=f'{current_dir}/inputs/list_sp_ml.csv', help='Path to the species list.')
parser.add_argument('--i', type=str, default=f'{current_dir}/audio/test_bugg', required=True, help='Input audio sample.')
parser.add_argument('--flist', type=str, default=None, help='Path to the filter list of species.')
parser.add_argument('--device_id', type=str, default=None, required=True, help='Device id.')
parser.add_argument('--country', type=str, default="unknown", help='Country')
parser.add_argument('--lat', type=float, default=None, help='Latitude for geographic filtering.')
parser.add_argument('--lng', type=float, default=None, help='Longitude for geographic filtering.')
parser.add_argument('--model_name', type=str, default='birdnet', help='Name of the model to use.')
parser.add_argument('--model_checkpoint', type=str, default=None, help='Model checkpoint.')


def run(args, id="wabad"):
    if not isinstance(args, PipelineSchema):
        args = PipelineSchema(**vars(args))

    # Run the algorithm
    _, predictions = run_algorithm(args, id)

    if predictions is None:
        print(f"Skipping audio file {args.i}")
        return None
    
    # Save results directly to parquet
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(args.i))[0]
    os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)
    output_path = f"{OUTPUT_DIR}/predictions/predictions_{filename}_{timestamp}_{args.device_id}.parquet"
    
    try:
        predictions.to_parquet(output_path, index=False)
        print(f"Successfully saved {len(predictions)} predictions to {output_path}")
        status = {
            "status": "success",
            "file": output_path,
            "records": len(predictions)
        }
    except Exception as e:
        print(f"Error saving predictions: {e}")
        # Save to failed directory as backup
        os.makedirs(f"{OUTPUT_DIR}/failed", exist_ok=True)
        fallback_path = f"{OUTPUT_DIR}/failed/{filename}_{timestamp}.parquet"
        predictions.to_parquet(fallback_path, index=False)
        status = {
            "status": "error",
            "file": fallback_path,
            "error": str(e)
        }
        
    return status


if __name__ == "__main__":
    result = run(parser.parse_args())
    if result:
        print(f"Status: {result['status']}")
    print(f"It took {(time.time() - start_time):.2f}s to analyze the audio files.")