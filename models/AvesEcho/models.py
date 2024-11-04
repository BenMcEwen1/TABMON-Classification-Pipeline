import os.path
from .hear21passt.base import get_basic_model, get_model_passt

import torch
import shutil
import logging
import librosa
from typing import Any, Optional
import json
import tempfile
import datetime
import pandas as pd
from multiprocessing import Pool
import soundfile as sf

from AvesEcho.dataset import InferenceDataset
from AvesEcho.inference import inference #, inference_maxpool

RANDOM_SEED = 1337
SAMPLE_RATE_AST = 32000
SAMPLE_RATE = 48000
SIGNAL_LENGTH = 3 # seconds
SPEC_SHAPE = (298, 128) # width x height
FMIN = 20
FMAX = 15000
MAX_AUDIO_FILES = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_species_list(path):
    """Load the species list from a file."""
    species_list = []
    with open(path) as file:
        for line in file:
            species_list.append(line.strip())
    return sorted(species_list)


def save_chunk(args):
    #Function to save a single chunk of audio data to a file.
    chunk, save_path, rate = args
    sf.write(save_path, chunk, rate)

def split_signals(filepath, output_dir, signal_length=15, n_processes=None):
    """
    Function to split an audio signal into chunks and save them using multiprocessing.
    
    Args:
    - filepath: Path to the input audio file.
    - output_dir: Directory where the output chunks will be saved.
    - signal_length: Length of each audio chunk in seconds.
    - n_processes: Number of processes to use in multiprocessing. If None, the number will be determined automatically.
    """
 
    # Configure logging
    logging.basicConfig(filename='audio_errors.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
    # print("here")
    try:
        # Load the signal
        sig, rate = librosa.load(filepath, sr=None, offset=0.0, duration=None, res_type='kaiser_fast')
    except Exception as e:
        logging.error(f"Error loading audio from {filepath}: {e}")
        return []

    # Split signal into chunks
    sig_splits = [sig[i:i + int(signal_length * rate)] for i in range(0, len(sig), int(signal_length * rate)) if len(sig[i:i + int(signal_length * rate)]) == int(signal_length * rate)]
    # print(sig_splits)
    # Prepare multiprocessing   
    with Pool(processes=n_processes) as pool:
        args_list = []
        for s_cnt, chunk in enumerate(sig_splits):
            save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_{s_cnt}.wav")
            args_list.append((chunk, save_path, rate))
        
        # Save each chunk in parallel
        pool.map(save_chunk, args_list)


class AvesEcho:
    def __init__(self, model_name, slist, flist, add_filtering, mconf, maxpool, add_csv, args, outputd, avesecho_mapping):
        self.slist = slist
        self.flist = flist
        self.model_name = model_name
        self.add_filtering = add_filtering
        self.add_csv = add_csv
        self.mconf = mconf
        self.outputd = outputd
        self.avesecho_mapping = avesecho_mapping
        self.species_list = load_species_list(self.slist)
        self.n_classes = len(self.species_list)
        self.split_signals = split_signals
        self.maxpool = maxpool
        self.args = args
        self.algorithm_mode = True # self.determine_algorithm_mode(args.algorithm_mode)

        # print()
        # print(f"Running AvesEcho-v1 in {self.algorithm_mode.value} mode.")
        # print()

        # Load the model
        if self.model_name == 'passt':
            # print("passt")
            self.model = get_basic_model(mode = 'logits', arch="passt_s_kd_p16_128_ap486") # "passt_s_kd_p16_128_ap486"
            self.model.net =  get_model_passt(arch="passt_s_kd_p16_128_ap486",  n_classes=self.n_classes)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load('AvesEcho/checkpoints/best_model_passt.pt', map_location=device))
        if self.model_name == 'fc':
            print("fc")
            self.model = avesecho(NumClasses=self.n_classes)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load('checkpoints/best_model_fc_1.pt', map_location=device))

    def analyze(self, audio_input: str, lat: Optional[Any]=None, lon: Optional[Any]=None, result_file: Optional[str]=None):
        # NOTE: only using directories mode for now
        if self.algorithm_mode == True:
            self.analyze_directories(audio_input, lat, lon, result_file)
        # else:
        #     app.run(
        #         debug=self.str2bool(os.getenv("DEBUG")),
        #         host=os.getenv("HOST", "0.0.0.0"),
        #         port=int(os.getenv("PORT", "5000")),
        #         use_reloader=False,
        #     )

    def analyze_directories(
        self, audio_input: str, lat: Optional[Any]=None, lon: Optional[Any]=None, result_file: Optional[str]=None
    ) -> None:
        if os.path.isfile(audio_input):
            audio_files = [audio_input]
        else:
            audio_files = [
                os.path.join(audio_input, filename)
                for filename in os.listdir(audio_input)
                if not self.ignore_filesystem_object(audio_input, filename)
            ]

        analysis_results, _ = self.analyze_audio_files(audio_files, lat, lon)

        if os.path.isfile(audio_input):
            # Make sure the result file has the .json extension
            json_filename = result_file if result_file.endswith('.json') else f'{result_file}.json'
            json_path = f'AvesEcho/outputs/{json_filename}'
        else:
            json_path = result_file or 'AvesEcho/outputs/analysis-results.json'

        # Write the analysis results to a JSON file
        with open(json_path, 'w') as json_file:
            json.dump(analysis_results, json_file, indent=4)

    def ignore_filesystem_object(self, directory: str, filename: str) -> bool:
        return os.path.isdir(os.path.join(directory, filename)) or filename.startswith(".")

    def analyze_audio_files(
        self, audio_files, lat: Optional[Any]=None, lon: Optional[Any]=None
    ) -> tuple[dict[str, Any], int]:

        # Running the model to get predictions, and then returning the results.

        # NOTE: This filters by location (lat, long), not being used currently
        filtering_list = None # setup_filtering(lat, lon, self.add_filtering, self.flist, self.slist)

        analysis_results = {
            "generated_by": {
                "datetime": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tag": "1fd68f8c8cb93ec4e45049fcf9a056628e9599aa815790a2a7b568aa",
                "version": "avesecho-v1"
            },
            "media": [],
            "region_groups": [],
            "predictions": []
        }

        supported_file_extensions = ['.wav', '.mp3', '.ogg', '.flac']

        # For endpoint mode: write the audio files to a temporary directory.
        with tempfile.TemporaryDirectory() as temporary_directory:
            for audio_file in audio_files:
                filename = os.path.basename(audio_file) if self.algorithm_mode == True else audio_file.filename
                file_extension = os.path.splitext(filename)[1].lower()

                if file_extension in supported_file_extensions:
                    if self.algorithm_mode == True:
                        audio_file_path = audio_file
                    else:
                        audio_file_path = os.path.join(temporary_directory, filename)
                        with open(audio_file_path, "wb") as output_file:
                            output_file.write(audio_file.read())

                    self.analyze_audio_file(audio_file_path, filtering_list, analysis_results)
                else:
                    print(f"Audio file '{filename}' has an unsupported extension (supported are: {supported_file_extensions}).")
                    return {"error": f"Audio file '{filename}' has an unsupported extension (supported are: {supported_file_extensions})."}, 415

        return analysis_results, 200

    def analyze_audio_file(self, audio_file_path: str, filtering_list: list[str], analysis_results: dict[str, Any]) -> None:
        if not os.path.exists(self.outputd):
            os.makedirs(self.outputd)

        # Load soundfile and split signal into 3s chunks
        self.split_signals(audio_file_path, self.outputd, signal_length=3, n_processes=10)

        # Extract the filename from the path
        filename = audio_file_path.split('/')[-1]  # This splits the string by '/' and gets the last element

        # Load a list of files for in a dir
        inference_dir = self.outputd

        # NOTE: Error here, file path not formatted correctly using os.path.join()
        inference_data = [
            (inference_dir + "/" + f) for f in sorted(os.listdir(inference_dir))
        ]

        # Inference
        inference_set = InferenceDataset(inference_data, self.n_classes, model=self.model_name)
        params_inf = {'batch_size': 64, 'shuffle': False, 'num_workers': 5}
        inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)

        # Maps species common names to scientific names and also across XC and eBird standards and codes
        # NOTE: This is not used currently
        # df = pd.read_csv(self.avesecho_mapping, header=None, names=['ScientificName', 'CommonName'])

        # if self.maxpool:
        #     predictions, scores, files = inference_maxpool(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
        #     print(predictions, scores)
        #     # create_json_maxpool(self.algorithm_mode, analysis_results, predictions, scores, files, self.args, df, self.add_csv, filename, self.mconf, len(inference_data))
        # else:
        predictions, scores, files = inference(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
        print(predictions, scores)
            # create_json(self.algorithm_mode, analysis_results, predictions, scores, files, self.args, df, self.add_csv, filename, self.mconf)

        # Empty temporary audio chunks directory
        # shutil.rmtree(self.outputd)



def run_algorithm(avesecho_mapping=None, result_file=None):
    global classifier

    classifier = AvesEcho(model_name="passt", slist='AvesEcho/inputs/list_sp_ml.csv', flist=None,
                          add_filtering=None, mconf=None,
                          outputd="AvesEcho/output", avesecho_mapping=avesecho_mapping,
                          maxpool=True, add_csv=False, args=None)

    classifier.analyze(audio_input="audio", lat=None, lon=None, result_file=result_file)