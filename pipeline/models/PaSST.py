from hear21passt.base import get_basic_model, get_model_passt
import torch
import warnings

from dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_species_list(path):
    # Load the species list from a file.
    species_list = []
    with open(path) as file:
        for line in file:
            species_list.append(line.strip())
    return sorted(species_list)

def adjusted_threshold(count, threshold):
    c = count / (count + 100)
    return threshold * c + 0.1 * (1 - c)


if __name__ == "__main__":
    species_list = load_species_list("./pipeline/inputs/list_sp_ml.csv")
    species_thresholds = pd.read_csv('./pipeline/inputs/species_thresholds_AvesEcho_1.csv')

    # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
    model = get_basic_model(mode="logits")

    # optional replace the transformer with one that has the required number of classes i.e. 50
    model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=len(species_list))

    # load the pre-trained model state dict
    state_dict = torch.load('./pipeline/checkpoints/best_model_passt.pt', map_location=device, weights_only=True)
    # load the weights into the transformer
    model.net.load_state_dict(state_dict, strict=False)

    softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    inference = Dataset("./audio/single/")

    for input, label in DataLoader(inference, batch_size=10, shuffle=True):
        with torch.no_grad():
            logits=model(input) 
            outputs = sigmoid(logits)

            sorted_df_thresholds = species_thresholds.sort_values(by='Scientific Name')
            # Apply the function to each row in the sorted DataFrame and create a tensor of adjusted thresholds
            adjusted_thresholds = sorted_df_thresholds.apply(lambda row: adjusted_threshold(row['Count'], row['Threshold']), axis=1)
            adjusted_thresholds_tensor = torch.tensor(adjusted_thresholds.values)

            # Apply the threshold to get predictions mask for the entire output
            predicted_mask = outputs > adjusted_thresholds_tensor.to(device) 
            print(predicted_mask)

            # indices = indices.tolist()
            # labels = [species_list[i] for i in indices]
            # print(labels)

            # sm = sm.tolist()
            # probs = [sm[i][j] for i, j in enumerate(indices)]
            # print(probs)