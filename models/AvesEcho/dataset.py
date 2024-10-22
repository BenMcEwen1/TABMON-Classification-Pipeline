import torch
import json
import librosa
import numpy as np
from torchaudio.transforms import Resample

SAMPLE_RATE = 48000
SAMPLE_RATE_AST = 32000

           
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, NumClasses, model):
        self.list_IDs = list_IDs
        self.NumClasses = NumClasses
        self.model = model

        with open('AvesEcho/inputs/global_parameters.json', 'r') as json_file:
            parameters = json.load(json_file)

        self.global_mean = parameters['global_mean']
        self.global_std = parameters['global_std']

    def __len__(self):
        # Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        ID = self.list_IDs[index]

        # Open the file with librosa (limited to the first certain number of seconds)
        x, rate = librosa.load(ID, sr=None, offset=0.0, res_type='kaiser_fast')

        x = (x - self.global_mean) / self.global_std
        #convert mixed to tensor
        x = torch.from_numpy(x).float() 

        resampler = Resample(rate, SAMPLE_RATE_AST, dtype=x.dtype)
        x = resampler(x)  
        birdnet_embedding = np.zeros(320)   

         
        return {'inputs': x, 'emb': birdnet_embedding, 'file': ID} 
    