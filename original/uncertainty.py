import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from models import avesecho, ContextAwareHead
from embeddings import Embeddings
from torch.nn.functional import sigmoid
import torch.nn as nn
from util import load_species_list
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
annotation_file = "../audio/sound-of-norway/annotation_split.csv"


def uncertaintyEntropy(outputs:torch.tensor, threshold:float=0.1):
    """
    Calculate aggregated uncertainty (entropy) for a batch of multi-label classes.
    Binary Entropy - https://en.wikipedia.org/wiki/Binary_entropy_function

    Parameters
    ----------
    outputs : tensor
        Output of the model
    threshold : float, optional
        Threshold for filtering low uncertainty values, by default 0.1

    Returns: List of uncertainties per batch
    """
    uncertainty = []
    for p in outputs:
        uncertainty_array = -(p*torch.log2(p))-(torch.subtract(1, p))*torch.log2(torch.subtract(1, p))
        uncertainty_array = torch.nan_to_num(uncertainty_array)
        # if threshold:
        #     filtered_uncertainty = [x for x in uncertainty_array if x > threshold]
        av_uncertainty = sum(uncertainty_array)/len(uncertainty_array)
        uncertainty.append(av_uncertainty.item())
    return uncertainty

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, emb):
        x = self.sigmoid(emb)
        x = self.fc1(emb.squeeze(1))
        return x


class AudioDataset(Dataset):
    # ANNOTATIONS, AUDIO_DIR, mel_spectrogram, 16000, False
    def __init__(self, annotations_file, audio_dir, val):
        super(AudioDataset, self).__init__()
        self.val = val
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        embedding = None

        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)

        path = os.path.dirname(audio_path)
        filename = os.path.basename(os.path.splitext(audio_path)[0].lower()) + '.pt'
        embedding_path = os.path.join(path, filename)

        if os.path.exists(embedding_path):
            embedding = torch.load(embedding_path, map_location=torch.device(device))
        else:
            pass

        return embedding, label, audio_path

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,19], self.annotations.iloc[index,16], self.annotations.iloc[index,2])
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,16]

    def _filter_annotations(self, data):
        return data.loc[(data['Data_split'] == self.val)]


def generate_uncertainty(n_heads, n_context, model=None, device='cpu'):
    dataset = AudioDataset(annotation_file, "../audio/sound-of-norway/", "*")

    species_list = load_species_list('inputs/list_en_ml.csv')

    df = pd.read_csv('../audio/sound-of-norway/annotation_adjusted.csv')
    df = df[df['Data_split'] != 0]
    labels = df['new_label'].unique().tolist()

    # Override to test only Sounds of Norway labels
    species_list = labels

    context = Embeddings('../audio/sound-of-norway/combined.bin')

    if model is None:
        print("Loading model...")
        model = ContextAwareHead(n_classes=len(species_list), heads=n_heads, context=n_context).to(device)
        
        # model = MLP(input_size=320, num_classes=len(species_list)).to(device)
        # model.load_state_dict(torch.load('./inputs/checkpoints/BirdNet_1.pt', map_location=device))
    
    uncertainty = []
    file_path = []

    model.eval()
    with torch.no_grad():
        print("Updating uncertainty values...")
        for embedding, _, audio_path in dataset:
            
            if embedding is None:
                uncertainty.append(None)
            else:
                embedding.to(device)

                # In-context sampling
                samples, _ = context.search_embeddings(embedding, k=n_context+1)
                samples = torch.tensor(samples, dtype=torch.float32).to(device)
                logits = model(embedding, samples)

                # logits = model(x=None, emb=embedding)
                outputs = sigmoid(logits)
                maxpool = torch.max(outputs, dim=0).values

                # Calculate uncertainty
                batch = embedding.shape[0] # Essential to scale by length
                uncertainty.append(uncertaintyEntropy(torch.unsqueeze(maxpool, 0))[0]/batch)
                file_path.append(audio_path)

    csv_path = "../audio/sound-of-norway/annotation_uncertainty.csv"
    df = pd.read_csv(csv_path)
    df['Uncertainty'] = pd.Series(uncertainty)
    df.to_csv(csv_path, index=False)
    return uncertainty

if __name__ == "__main__":
    annotation_file = "../audio/sound-of-norway/annotation_split.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = AudioDataset(annotation_file, "../audio/sound-of-norway/", "*")
    uncertainty = generate_uncertainty(dataset, device=device)

    # Add to dataframe
    csv_path = "../audio/sound-of-norway/annotation_uncertainty.csv"
    df = pd.read_csv(csv_path)
    df['Uncertainty'] = pd.Series(uncertainty)
    df.to_csv(csv_path, index=False)