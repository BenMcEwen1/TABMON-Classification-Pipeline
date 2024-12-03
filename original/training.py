import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from models import avesecho
from embeddings import embed
from torch.nn.functional import sigmoid
from util import load_species_list
from tqdm import tqdm
import wandb


annotation_file = "../audio/annotation_subset.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AudioDataset(Dataset):
    # ANNOTATIONS, AUDIO_DIR, mel_spectrogram, 16000, False
    def __init__(self, annotations_file, audio_dir, val):
        super(AudioDataset, self).__init__()
        self.val = val
        self.annotations = self._filter_annotations(pd.read_csv(annotations_file))
        print(self.annotations)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)

        path = os.path.dirname(audio_path)
        filename = os.path.basename(os.path.splitext(audio_path)[0].lower()) + '.pt'
        embedding_path = os.path.join(path, filename)

        if os.path.exists(embedding_path):
            embedding = torch.load(embedding_path)
        else:
            print(f"Generating embedding for {audio_path}")
            embedding = embed(audio_path)

        return embedding, label

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,3], self.annotations.iloc[index,2])
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,3] # Same as directory in this case

    def _filter_annotations(self, data):
        return data.loc[data['Data_split'] == self.val]


def train(dataset, validation, epochs=10, lr=0.001):
    wandb.init(
        project="Sounds of Norway (Benchmark)",

        config={
            "learning_rate": lr,
            "architecture": "AvesEcho (fc)",
            "dataset": "Sounds of Norway (Benchmark)",
            "epochs": epochs,
        }
    )
    
    species_list = load_species_list('inputs/list_en_ml.csv')

    model = avesecho(NumClasses=len(species_list)).to(device)
    model.load_state_dict(torch.load('./inputs/checkpoints/best_model_fc_1.pt', map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cross_entropy = torch.nn.CrossEntropyLoss()

    # training = DataLoader(dataset, batch_size=16, shuffle=True) # Only works for tensor with the same shape, another reason to have consistent audio segments

    def encode_label(labels):
        one_hot = torch.zeros(len(species_list))    

        for label in labels:
            index = species_list.index(label.replace(" ", ""))
            one_hot[index] = 1
        return one_hot

    # TODO: Add dataloader
    for epoch in range(epochs):

        training_loss = 0
        for embedding, label in dataset:
            print(embedding.shape)
            embedding.to(device)
            model.train()
            optimizer.zero_grad()

            # Convert label to one-hot vector
            y = encode_label([label]).to(device)

            logits = model(x=None, emb=embedding)
            outputs = sigmoid(logits)
            maxpool = torch.max(outputs, dim=0).values

            loss = cross_entropy(maxpool, y)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for embedding, label in validation:
                embedding.to(device)
                y = encode_label([label]).to(device)

                logits = model(x=None, emb=embedding)
                outputs = sigmoid(logits)
                maxpool = torch.max(outputs, dim=0).values

                loss = cross_entropy(maxpool, y)
                val_loss += loss.item()

        print(f"Epoch: {epoch}: Training loss: {training_loss}, Validation loss: {val_loss}")
        wandb.log({"Training loss": training_loss, "Validation loss": val_loss})
        

if __name__ == "__main__":
    dataset = AudioDataset(annotation_file, "../audio/SoN/Train/", "Train")
    val = AudioDataset(annotation_file, "../audio/SoN/Test/", "Test")

    train(dataset, val)