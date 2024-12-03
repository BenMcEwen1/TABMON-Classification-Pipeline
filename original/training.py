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
            embedding = torch.load(embedding_path, map_location=torch.device(device))
        else:
            print(f"Generating embedding for {audio_path}")
            embedding = embed(audio_path)

        return embedding, label

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,16], self.annotations.iloc[index,2])
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,16]

    def _filter_annotations(self, data):
        return data.loc[data['Data_split'] == self.val]


def train(dataset, validation, epochs=100, lr=0.001, device='cpu'):
    wandb.init(
        project="Sounds of Norway (Benchmark) - Server",

        config={
            "learning_rate": lr,
            "architecture": "AvesEcho (fc)",
            "dataset": "Sounds of Norway (Benchmark)",
            "epochs": epochs,
        }
    )
    
    species_list = load_species_list('inputs/list_en_ml.csv')

    model = avesecho(NumClasses=len(species_list)).to(device)
    #model.load_state_dict(torch.load('./inputs/checkpoints/best_model_fc_1.pt', map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cross_entropy = torch.nn.CrossEntropyLoss()

    # training = DataLoader(dataset, batch_size=16, shuffle=True) # Only works for tensor with the same shape, another reason to have consistent audio segments

    def encode_label(labels):
        one_hot = torch.zeros(len(species_list))
        #print(f"True label: {labels[0]}")

        for label in labels:
            if label.replace(" ", "") not in species_list:
                # Actual labels: ZZZ_Unsure, ZZZ_Unsure: Not bird,  ZZZ_Unsure: Other bird 
                index = species_list.index("Unsure")
                one_hot[index] = 1
                #print(f"Species List: {species_list[index]}")
            else:
                index = species_list.index(label.replace(" ", ""))
                one_hot[index] = 1
                #print(f"Species List: {species_list[index]}")
        #print("---")
        return one_hot

    # TODO: Add dataloader
    for epoch in range(epochs):

        training_loss = 0
        for ti, (embedding, label) in enumerate(dataset):
            embedding.to(device)
            model.train()
            optimizer.zero_grad()

            # Convert label to one-hot vector
            y = encode_label([label]).to(device)

            logits = model(x=None, emb=embedding)
            outputs = sigmoid(logits)
            maxpool = torch.max(outputs, dim=0).values

            # Sanity Check
            if ti in torch.randint(0, 1000, (10,)).tolist():
                print(f"True Label: {label} | Predicted label: {species_list[maxpool.argmax().item()]}")

            loss = cross_entropy(maxpool, y)
            training_loss += loss.item()/(ti + 1)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for vi, (embedding, label) in enumerate(validation):
                embedding.to(device)
                y = encode_label([label]).to(device)

                logits = model(x=None, emb=embedding)
                outputs = sigmoid(logits)
                maxpool = torch.max(outputs, dim=0).values

                loss = cross_entropy(maxpool, y)
                val_loss += loss.item()/(vi + 1)

        print(f"Epoch: {epoch}: Training loss: {training_loss}, Validation loss: {val_loss}")
        wandb.log({"Training loss": training_loss, "Validation loss": val_loss})

    torch.save(model.state_dict(), f'./inputs/checkpoints/SoN_{epoch}.pt')

if __name__ == "__main__":
    annotation_file = "../benchmark_sound-of-norway/annotation_split.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = AudioDataset(annotation_file, "../benchmark_sound-of-norway/Train/", "Train")
    val = AudioDataset(annotation_file, "../benchmark_sound-of-norway/Test/", "Test")

    train(dataset, val, device=device)
