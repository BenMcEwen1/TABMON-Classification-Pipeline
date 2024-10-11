import torch 
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

# Aim is to seperate audio files into 3 seconds chunks

audio_path = "./audio/2022-09-24T02_03_08.447Z.mp3"

signal, sr = torchaudio.load(audio_path)
print(signal)
print(sr)

print(len(signal[0]) % sr) # Check to confirm signal divible by sample rate

# Resample
def resample(signal, sr, new_sr):
    resample = torchaudio.transforms.Resample(sr, new_sr)
    signal = resample(signal[0])
    return signal

audio_chunks = []
for i in range(0, len(signal[0]), sr*3):
    audio_chunks.append(signal[0][i:i+sr*3])

print(len(audio_chunks))

class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, sr, val):
        super(AudioDataset, self).__init__()
        self.val = val
        # self.annotations = self._filter_annotations(pd.read_csv(annotations_file))
        self.audio_dir = audio_dir
        self.target_rate = sr
        # self.resize = transforms.Resize((224,224))
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        # signal, sr = torchaudio.load(audio_path)
        # resample = torchaudio.transforms.Resample(sr, self.target_rate)
        # signal = resample(signal[0])
        # feature = self.feature_extractor(signal, sampling_rate=self.target_rate, return_tensors="pt")
        # feature = feature['input_values']
        return 1,1 #feature,label

    # def _get_audio_path(self, index):
    #     path = os.path.join(self.audio_dir, self.annotations.iloc[index,1])
    #     return path

    # def _get_audio_label(self, index):
    #     return self.annotations.iloc[index,3]

    # def _get_validation(self, index):
    #     return self.annotations.iloc[index,2]

    # def _filter_annotations(self, data):
    #     return data.loc[data['Validation'] == self.val]
    
    # def shuffle(self):
    #     pass

    # def split(self, audio, sr):
    #     pass


