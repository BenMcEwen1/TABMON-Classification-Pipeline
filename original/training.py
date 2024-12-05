import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from models import avesecho
from embeddings import embed
from torch.nn.functional import sigmoid
import torch.nn as nn
from util import load_species_list
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import wandb


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

        return embedding, label, audio_path

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,16], self.annotations.iloc[index,2])
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,16]

    def _filter_annotations(self, data):
        return data.loc[data['Data_split'] == self.val]


def train(dataset, validation, epochs=1, lr=0.001, device='cpu', eval=True, average='macro'):
    wandb.init(
        project="AvesEcho (fc) Finetuned",
        config={
            "learning_rate": lr,
            "architecture": "AvesEcho + MLP",
            "dataset": "Sounds of Norway (Benchmark)",
            "epochs": epochs,
        }
    )
    
    species_list = load_species_list('inputs/list_en_ml.csv')

    df = pd.read_csv('../audio/benchmark_sound-of-norway/annotation_adjusted.csv')
    df = df[df['Data_split'] != 0]
    labels = df['new_label'].unique().tolist()

    # Override to test only Sounds of Norway labels
    # species_list = labels
    # print(species_list)

    model = avesecho(NumClasses=585).to(device)
    model.load_state_dict(torch.load('./inputs/checkpoints/avesecho_fc_SoN.pt', map_location=device))

    # model = MLP(input_size=320, num_classes=len(species_list)).to(device)
    # model.load_state_dict(torch.load('./inputs/checkpoints/BirdNet_filtered_labels.pt', map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cross_entropy = torch.nn.BCELoss()

    def encode_label(labels):
        one_hot = torch.zeros(len(species_list))
        for label in labels:
            if label.replace(" ", "") not in species_list:
                # Actual labels: ZZZ_Unsure, ZZZ_Unsure: Not bird,  ZZZ_Unsure: Other bird 
                index = species_list.index("Unsure")
                one_hot[index] = 1
            else:
                index = species_list.index(label.replace(" ", ""))
                one_hot[index] = 1
        return one_hot
    
    y_pred = []
    y_true = []
    uncertainty = []
    file_path = []

    # TODO: Add dataloader
    for epoch in range(epochs):
        if not eval:
            training_loss = 0
            for ti, (embedding, label) in enumerate(dataset):
                embedding.to(device)
                model.train()
                optimizer.zero_grad()

                # Convert label to one-hot vector
                y = encode_label([label]).to(device)

                logits = model(x=None, emb=embedding)
                # print(logits.shape)
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
            for vi, (embedding, label, audio_path) in enumerate(validation):
                embedding.to(device)
                y = encode_label([label]).to(device)

                logits = model(x=None, emb=embedding)
                outputs = sigmoid(logits)
                maxpool = torch.max(outputs, dim=0).values

                # Calculate uncertainty
                uncertainty.append(uncertaintyEntropy(torch.unsqueeze(maxpool, 0))[0])
                file_path.append(audio_path)

                loss = cross_entropy(maxpool, y)
                val_loss += loss.item()/(vi + 1)

                binary_tensor = torch.zeros(585)
                binary_tensor[maxpool.argmax().item()] = 1

                y_pred.append(species_list[maxpool.argmax().item()])
                y_true.append(label)

                labels = [label.replace(" ", "") for label in labels]

            if eval:
                y_pred = ["Unsure" if (label=="Other Bird" or label=="Not Bird" or label=="Multi Bird") else label for label in y_pred]
                y_true = ["Unsure" if (label=="Other Bird" or label=="Not Bird" or label=="Multi Bird") else label for label in y_true]
                y_true = [label.replace(" ", "") for label in y_true]
                # labels = [label for label in labels if (label != "ZZZ_Unsure" and label != "ZZZ_Multi" and label != "Not Bird" and label != "Other Bird" and label != "Eurasian Curlew" )]
                labels = [label for label in labels if (label != "Unsure" and label != "NotBird" and label != "MultiBird" and label != "OtherBird" and label != "EurasianCurlew" and label != "CommonBuzzard" )]
                
                labels.sort()
                labels.append("Unsure")

                print(y_true)


                print(f"Precision ({average}): {precision_score(y_true, y_pred, average=average)}")
                print(f"Recall ({average}): {recall_score(y_true, y_pred, average=average)}")
                print(f"F1 Score ({average}): {f1_score(y_true, y_pred, average=average)}")

                # Plot confusion matrix
                cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=labels)
                disp.plot(xticks_rotation='vertical', text_kw={'fontsize': 6})
                plt.tick_params(labelsize=6)
                plt.show()

    #     print(f"Epoch: {epoch}: Training loss: {training_loss}, Validation loss: {val_loss}")
    #     wandb.log({"Training loss": training_loss, "Validation loss": val_loss})

    # torch.save(model.state_dict(), f'./inputs/checkpoints/BirdNet_{epoch}.pt')


    # uncertainty = [i/max(uncertainty) for i in uncertainty]

    # mapping = {k: i for i, k in enumerate(sorted(uncertainty, reverse=True))}
    # print(*mapping.keys())

    # for i in range(len(uncertainty)):
    #     print(f"File: {file_path[i]}, Uncertainty: {uncertainty[i]}, Label: {y_true[i]}, Prediction: {y_pred[i]}")



if __name__ == "__main__":
    annotation_file = "../audio/benchmark_sound-of-norway/annotation_split.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = AudioDataset(annotation_file, "../audio/benchmark_sound-of-norway/Train/", "Train")
    val = AudioDataset(annotation_file, "../audio/benchmark_sound-of-norway/Test/", "Test")

    train(dataset, val, device=device)
