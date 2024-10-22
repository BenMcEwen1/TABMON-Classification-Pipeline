import torch
import librosa
from torchaudio.transforms import Resample
from os import listdir
from os.path import isfile, join

from helpers import split_signals

SAMPLE_RATE_AST = 32000

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory='./'):
        self.directory = directory
        self.temp = "./audio/temp/"
        
        split_signals(directory, self.temp, signal_length=3, n_processes=10)
        self.files = [join(self.temp, f) for f in listdir(self.temp) if isfile(join(self.temp, f))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio, rate = librosa.load(self.files[index], sr=None, res_type='kaiser_fast')
        
        audio = torch.from_numpy(audio).float() 
        resampler = Resample(rate, SAMPLE_RATE_AST, dtype=audio.dtype)
        audio = resampler(audio)  

        return audio, "Bird"