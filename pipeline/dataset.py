# Adapted from the AvesEcho repository (https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1)
# Original author: Burooj Ghani
# Adapted by: Ben McEwen

import torch
import librosa
from torchaudio.transforms import Resample
from os import listdir
from os.path import isfile, join
import json
import numpy as np

from helpers import split_signals
from models.birdnet import BirdNet, embed_sample

SAMPLE_RATE = 32000
SAMPLE_RATE_BIRDNET = 48000

birdnet_weights = './pipeline/checkpoints/BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite'

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, NumClasses, model):
        self.list_IDs = list_IDs
        self.NumClasses = NumClasses
        self.embedding_model = BirdNet(SAMPLE_RATE_BIRDNET, birdnet_weights)
        self.model = model

        with open('./pipeline/inputs/global_parameters.json', 'r') as json_file:
            parameters = json.load(json_file)

        self.global_mean = parameters['global_mean']
        self.global_std = parameters['global_std']

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        file = self.list_IDs[index]

        # Open the file with librosa (limited to the first certain number of seconds)
        try:
            x, _ = librosa.load(file, sr=SAMPLE_RATE)
        except:
            x, _ = [], SAMPLE_RATE   

        x = (x - self.global_mean) / self.global_std
        # Convert mixed to tensor
        x = torch.from_numpy(x).float() 

        if self.model == 'passt':
            # Resample the audio from 48k to 32k for PaSST
            resampler = Resample(SAMPLE_RATE_BIRDNET, SAMPLE_RATE, dtype=x.dtype)
            x = resampler(x)  
            birdnet_embedding = np.zeros(320) 
        else:
            try:
                # Compute BirdNET embedding
                outputs, _ = embed_sample(self.embedding_model, x.numpy(), SAMPLE_RATE_BIRDNET)
                birdnet_embedding = np.expand_dims(outputs, axis=0)
            except:
                print("BirdNET embedding failed")
                birdnet_embedding = np.zeros(320)  

        return {'inputs': x, 'emb': birdnet_embedding, 'file': file} 

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, directory='./'):
#         self.directory = directory
#         self.temp = "./audio/temp/"
        
#         split_signals(directory, self.temp, signal_length=3, n_processes=10)
#         self.files = [join(self.temp, f) for f in listdir(self.temp) if isfile(join(self.temp, f))]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         audio, rate = librosa.load(self.files[index], sr=None, res_type='kaiser_fast')
        
#         audio = torch.from_numpy(audio).float() 
#         resampler = Resample(rate, SAMPLE_RATE, dtype=audio.dtype)
#         audio = resampler(audio)  

#         return audio, "Bird"