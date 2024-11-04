import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import shutil
from typing import Optional, Any
import tempfile
import json
import pandas as pd
from enum import Enum
import datetime

from helpers import split_signals, load_species_list, setup_filtering, create_json, create_json_maxpool
from dataset import InferenceDataset
from inference import inference, inference_maxpool

class AlgorithmMode(Enum):
    DIRECTORIES = "directories"
    ENDPOINT = "endpoint"

n_classes = 585

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class avesecho(nn.Module):
    def __init__(self, NumClasses=n_classes, pretrain=True, ExternalEmbeddingSize=320, hidden_layer_size=100):
        super(avesecho, self).__init__()
        self.fc1 = nn.Linear(ExternalEmbeddingSize, NumClasses)

    def forward(self, x, emb):
        x = self.fc1(emb.squeeze(1))
        return x  

class AvesEcho:
    def __init__(self, model_name, slist, flist, add_filtering, mconf, maxpool, add_csv, outputd, avesecho_mapping):
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

        # Load the model
        # if self.model_name == 'passt':
        #     self.model = get_basic_model(mode = 'logits', arch="passt_s_kd_p16_128_ap486")
        #     self.model.net =  get_model_passt(arch="passt_s_kd_p16_128_ap486",  n_classes=self.n_classes)
        #     self.model = self.model.to(device)
        #     self.model.load_state_dict(torch.load('checkpoints/best_model_passt.pt', map_location=device))
        if self.model_name == 'fc':
            self.model = avesecho(NumClasses=self.n_classes)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load('./pipeline/checkpoints/best_model_fc_1.pt', map_location=device))

    def analyze(self, audio_input: str, lat: Optional[Any]=None, lon: Optional[Any]=None, result_file: Optional[str]=None):
        self.analyze_directories(audio_input, lat, lon, result_file)

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
            json_path = f'outputs/{json_filename}'
        else:
            json_path = result_file or 'outputs/analysis-results.json'

        # Write the analysis results to a JSON file
        with open(json_path, 'w') as json_file:
            json.dump(analysis_results, json_file, indent=4)

    def ignore_filesystem_object(self, directory: str, filename: str) -> bool:
        return os.path.isdir(os.path.join(directory, filename)) or filename.startswith(".")

    def analyze_audio_files(
        self, audio_files: list, lat: Optional[Any]=None, lon: Optional[Any]=None
    ) -> tuple[dict[str, Any], int]:
        # Running the model to get predictions, and then returning the results.

        filtering_list = setup_filtering(lat, lon, self.add_filtering, self.flist, self.slist)

        analysis_results = {
            "generated_by": {
                "datetime": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tag": "1fd68f8c8cb93ec4e45049fcf9a056628e9599aa815790a2a7b568aa",
                "version": "avesecho-v1 1.2.0"
            },
            "media": [],
            "region_groups": [],
            "predictions": []
        }

        supported_file_extensions = ['.wav', '.mp3', '.ogg', '.flac']

        # For endpoint mode: write the audio files to a temporary directory.
        with tempfile.TemporaryDirectory() as temporary_directory:
            for audio_file in audio_files:
                filename = os.path.basename(audio_file) # if self.algorithm_mode == AlgorithmMode.DIRECTORIES else audio_file.filename
                file_extension = os.path.splitext(filename)[1].lower()

                if file_extension in supported_file_extensions:
                    audio_file_path = audio_file
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
        inference_data = [
            os.path.join(inference_dir, f)
            for f in sorted(os.listdir(inference_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        ]

        # Inference
        inference_set = InferenceDataset(inference_data, self.n_classes, model=self.model_name)
        params_inf = {'batch_size': 64, 'shuffle': False, 'num_workers': 0} # NOTE: num_workers was 5, issue with multi-processing on Windows
        inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)

        # Maps species common names to scientific names and also across XC and eBird standards and codes
        df = pd.read_csv(self.avesecho_mapping, header=None, names=['ScientificName', 'CommonName'])

        if self.maxpool:
            predictions, scores, files = inference_maxpool(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
            create_json_maxpool(analysis_results, predictions, scores, files, self.flist, df, self.add_csv, filename, self.mconf, len(inference_data))
        else:
            predictions, scores, files = inference(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
            create_json(analysis_results, predictions, scores, files, self.flist, df, self.add_csv, filename, self.mconf)

        # Empty temporary audio chunks directory
        shutil.rmtree(self.outputd)
        return predictions, scores, files


    # def determine_algorithm_mode(self, algorithm_mode: str) -> AlgorithmMode:
    #     """Determine the mode ("directories" or "endpoint") the algorithm will run in. First the "--algorithm-mode"
    #        command-line argument is used to determine the algorithm mode. If that is not specified, the "ALGORITHM_MODE"
    #        environment variable determines the mode. If both are not specified, algorithm mode "directories" is used by
    #        default."""
    #     try:
    #         result = AlgorithmMode(algorithm_mode)

    #     except ValueError:
    #         print(f"Unknown algorithm mode '{algorithm_mode}' - defaulting to 'directories' mode.")
    #         result = AlgorithmMode.DIRECTORIES

    #     return result

    # def str2bool(self, value) -> bool:
    #     """
    #     Converts a string to boolean by matching it with some well known representations of bool.
    #     {"yes", "true", "t", "y", "1"} --> true
    #     If value is not one of these representations it returns False.
    #     :param value: a string representing a boolean value
    #     :return: a boolean value
    #     """
    #     return str(value).lower() in {"yes", "true", "t", "y", "1"}
