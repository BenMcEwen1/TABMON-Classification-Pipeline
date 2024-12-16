import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from models import avesecho, ContextAwareHead
from embeddings_old import embed
from embeddings import Database
from torch.nn.functional import sigmoid
import torch.nn as nn
from util import load_species_list
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from uncertainty import generate_uncertainty
from types import SimpleNamespace
import wandb

torch.autograd.set_detect_anomaly(True)

annotation_file = "../audio/sound-of-norway/annotation_uncertainty.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        av_uncertainty = sum(uncertainty_array)/len(uncertainty_array)
        uncertainty.append(av_uncertainty.item())
    return uncertainty

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, emb):
        x = self.fc1(emb.squeeze(1))
        return x


class AudioDataset(Dataset):
    # ANNOTATIONS, AUDIO_DIR, mel_spectrogram, 16000, False
    def __init__(self, annotations_file, audio_dir, val, subset=False, shuffle=False, uncertainty=False):
        super(AudioDataset, self).__init__()
        self.val = val
        self.annotations = self._filter_annotations(pd.read_csv(annotations_file))

        if shuffle:
            self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)
        elif uncertainty:
            self.annotations = self.annotations.sort_values(by=['Uncertainty'], ascending=False)

        if subset:
            self.annotations = self.annotations[:600] # Subset
        
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


def train(log=True, plot=True, resample=False, save=False, average='macro', min_loss=100000, config=None):
    dataset = AudioDataset(annotation_file, "../audio/sound-of-norway/Train/", "Train", shuffle=True, subset=False)
    validation = AudioDataset(annotation_file, "../audio/sound-of-norway/Test/", "Test")
    
    with wandb.init(project="Uncertainty Sampling Comparison", config=config):
        config = wandb.config

        species_list = load_species_list('inputs/list_en_ml.csv')

        df = pd.read_csv('../audio/sound-of-norway/annotation_adjusted.csv')
        df = df[df['Data_split'] != 0]
        labels = df['new_label'].unique().tolist()

        # Override to test only Sounds of Norway labels
        species_list = labels

        # model = ContextAwareHead(n_classes=len(species_list), heads=config.heads, context=config.context).to(device)
        # context = Database('../audio/sound-of-norway/combined.bin')
        # model.load_state_dict(torch.load('./inputs/checkpoints/Context_jhpq1ily.pt', map_location=device))

        # model = avesecho(NumClasses=585).to(device)
        # model.load_state_dict(torch.load('./inputs/checkpoints/avesecho_fc_SoN.pt', map_location=device))

        model = MLP(input_size=320, num_classes=len(species_list)).to(device)
        model.load_state_dict(torch.load('./inputs/checkpoints/BirdNet_49.pt', map_location=device))

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        cross_entropy = torch.nn.BCELoss()

        def encode_label(labels):
            one_hot = torch.zeros(len(species_list))
            for label in labels:
                if label not in species_list: 
                    index = species_list.index("Unsure")
                    one_hot[index] = 1
                else:
                    index = species_list.index(label)
                    one_hot[index] = 1
            return one_hot

        for epoch in range(config.epochs):
            training_loss = 0
            increment = 20

            if resample:
                dataset = AudioDataset(annotation_file, "../audio/sound-of-norway/Train/", "Train", shuffle=False, uncertainty=True)

            for ti, (embedding, label, _) in enumerate(dataset):
                if ti >= (epoch+1)*increment:
                    break

                model.train()
                embedding.to(device)
                optimizer.zero_grad()

                # Convert label to one-hot vector
                y = encode_label([label]).to(device)

                # In-context sampling
                # samples, _ = context.search_embeddings(embedding, k=config.context+1)
                # samples = torch.tensor(samples, dtype=torch.float32).to(device)
                # logits = model(embedding, samples)

                logits = model(x=None, emb=embedding)
                outputs = sigmoid(logits)
                maxpool = torch.max(outputs, dim=0).values

                loss = cross_entropy(maxpool, y)
                training_loss += loss.item()/(ti + 1)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = 0
                y_pred = []
                y_true = []
                uncertainty = []
                for vi, (embedding, label, _) in enumerate(validation):
                    embedding.to(device)
                    y = encode_label([label]).to(device)

                    # In-context sampling
                    # samples, _ = context.search_embeddings(embedding, k=config.context+1)
                    # samples = torch.tensor(samples, dtype=torch.float32).to(device)
                    # logits = model(embedding, samples)

                    logits = model(x=None, emb=embedding)
                    outputs = sigmoid(logits)
                    maxpool = torch.max(outputs, dim=0).values

                    # Sanity Check
                    if vi in torch.randint(0, 200, (2,)).tolist():
                        print(f"Validation: True Label: {label} | Predicted label: {species_list[maxpool.argmax().item()]}")

                    # Calculate uncertainty
                    batch = embedding.shape[0]
                    uncertainty.append(uncertaintyEntropy(torch.unsqueeze(maxpool, 0))[0]/batch)

                    loss = cross_entropy(maxpool, y)
                    val_loss += loss.item()/(vi + 1)

                    y_pred.append(species_list[maxpool.argmax().item()])
                    y_true.append(label)

            if log:
                print(f"Epoch: {epoch}: Training loss: {training_loss}, Validation loss: {val_loss}")
                wandb.log({"val_loss": val_loss, "Total Uncertainty": sum(uncertainty), "F1 Score": f1_score(y_true, y_pred, average=average)})

                if val_loss < min_loss:
                    best_epoch = epoch
                    best_model = model
                    min_loss = val_loss

            if resample:
                generate_uncertainty(n_heads=config.heads, n_context=config.context, model=model)

        if plot:
            y_pred = ["Unsure" if (label=="Other Bird" or label=="Not Bird" or label=="Multi Bird") else label for label in y_pred]
            y_true = ["Unsure" if (label=="Other Bird" or label=="Not Bird" or label=="Multi Bird") else label for label in y_true]
            labels = [label for label in labels if (label != "Unsure" and label != "Not Bird" and label != "Multi Bird" and label != "Other Bird" and label != "Eurasian Curlew" and label != "Common Buzzard" )]
            
            labels.sort()
            labels.append("Unsure")

            print(f"Precision ({average}): {precision_score(y_true, y_pred, average=average)}")
            print(f"Recall ({average}): {recall_score(y_true, y_pred, average=average)}")
            print(f"F1 Score ({average}): {f1_score(y_true, y_pred, average=average)}")

            # Plot confusion matrix
            cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=labels)
            disp.plot(xticks_rotation='vertical', text_kw={'fontsize': 4})
            plt.tick_params(labelsize=4)
            plt.show()

        if save:
            torch.save(best_model.state_dict(), f'./inputs/checkpoints/ContextOffline_{wandb.run.id}_{best_epoch}.pt')


if __name__ == "__main__":
    config = {
        "epochs": 1,
        "lr": 0.001,
        "context": 2,
        "heads": 2,
        "weight_decay": 0.00001,
        "save": False
    }

    config = SimpleNamespace(**config)
    train(config=config)


    # # Define sweep config
    # context_sweep = {
    #     "name": "Context (offline)",
    #     "method": "random",
    #     "metric": {"goal": "minimize", "name": "val_loss"},
    #     "parameters": {
    #         "epochs": {"value": 10},
    #         "heads": {"value": 2},
    #         "lr": {"value": 0.001},
    #         "context": {"values": [0, 1, 2, 4]},
    #         "weight_decay": {"value": 0.00001} # Definitely 0.00001
    #     },
    # }

    # # Initialize sweep by passing in config.
    # sweep_id = wandb.sweep(sweep=context_sweep, project="Context-model sweep")
    # wandb.agent(sweep_id, function=train)