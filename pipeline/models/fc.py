# Testing of AvesEcho CNN-based model - EfficentNet model (like BirdNet) trained on Xeno-canto (European bird) data 

import torch
import torch.nn as nn
import pandas as pd
from dataset import Dataset
from torch.utils.data import DataLoader

from helpers import load_species_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adjusted_threshold(count, threshold):
    c = count / (count + 100)
    return threshold * c + 0.1 * (1 - c)

species_thresholds = pd.read_csv('./pipeline/inputs/species_thresholds_AvesEcho_1.csv')
species_list = load_species_list('./pipeline/inputs/list_sp_ml.csv')
n_classes = n_classes=len(species_list)

class avesecho(nn.Module):
    def __init__(self, NumClasses=n_classes, pretrain=True, ExternalEmbeddingSize=320, hidden_layer_size=100):
        super(avesecho, self).__init__()
        self.fc1 = nn.Linear(ExternalEmbeddingSize, NumClasses)

    def forward(self, x, emb):
        x = self.fc1(emb.squeeze(1))
        return x    


model = avesecho(NumClasses=n_classes)
model = model.to(device)
model.load_state_dict(torch.load('./pipeline/checkpoints/best_model_fc_1.pt', map_location=device))
print(model)

# sigmoid = torch.nn.Sigmoid()
# inference = Dataset("audio/single/")

# for input, label in DataLoader(inference, batch_size=10, shuffle=True):
#     print(input.shape)
    
#     with torch.no_grad():
#         logits=model(input) 
#         outputs = sigmoid(logits)

#         sorted_df_thresholds = species_thresholds.sort_values(by='Scientific Name')
#         # Apply the function to each row in the sorted DataFrame and create a tensor of adjusted thresholds
#         adjusted_thresholds = sorted_df_thresholds.apply(lambda row: adjusted_threshold(row['Count'], row['Threshold']), axis=1)
#         adjusted_thresholds_tensor = torch.tensor(adjusted_thresholds.values)

#         # Apply the threshold to get predictions mask for the entire output
#         predicted_mask = outputs > adjusted_thresholds_tensor.to(device) 
#         print(predicted_mask)