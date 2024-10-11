import torch
import librosa
from torchaudio.transforms import Resample
from os import listdir
from os.path import isfile, join

SAMPLE_RATE_AST = 32000

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory='./'):
        self.directory = directory
        self.files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio, rate = librosa.load(self.files[index], sr=None, res_type='kaiser_fast')
        
        audio = torch.from_numpy(audio).float() 
        resampler = Resample(rate, SAMPLE_RATE_AST, dtype=audio.dtype)
        audio = resampler(audio)  

        return audio, "A bird!"