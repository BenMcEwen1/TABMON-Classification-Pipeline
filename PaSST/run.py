from hear21passt.base import get_basic_model, get_model_passt
import torch
import warnings

from dataset import Dataset
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_species_list(path):
    # Load the species list from a file.
    species_list = []
    with open(path) as file:
        for line in file:
            species_list.append(line.strip())
    return sorted(species_list)

# get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
model = get_basic_model(mode="logits")

# optional replace the transformer with one that has the required number of classes i.e. 50
model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=len(load_species_list("AvesEcho\inputs\list_sp_ml.csv")))

# load the pre-trained model state dict
state_dict = torch.load('AvesEcho/checkpoints/best_model_passt.pt', map_location=device)
# load the weights into the transformer
model.net.load_state_dict(state_dict, strict=False)


interference = Dataset("audio/")

for input, label in DataLoader(interference, batch_size=2, shuffle=True):
    print(input, label)

# Example inference 
# with torch.no_grad():
#     # audio_wave has the shape of [batch, seconds*32000] sampling rate is 32k
#     # example audio_wave of batch=3 and 10 seconds
#     audio = torch.ones((3, 32000 * 3))*0.5
#     audio_wave = audio.cpu()
#     logits=model(audio_wave) 