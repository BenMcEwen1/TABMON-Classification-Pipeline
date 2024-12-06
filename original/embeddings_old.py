import os
import argparse
from algorithm_mode import AlgorithmMode
from models import AvesEcho

default_algorithm_mode = os.getenv("ALGORITHM_MODE", AlgorithmMode.DIRECTORIES.value)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slist', type=str, default='./inputs/list_sp_ml.csv', help='Path to the species list.')
parser.add_argument('--flist', type=str, default='./inputs/species_list_nl.csv', help='Path to the filter list of species.')
parser.add_argument('--i', type=str, default='audio/NH-11_20240415_062840.WAV', help='Input audio sample.')
parser.add_argument("--algorithm_mode", default=default_algorithm_mode, help="Use input/output directories or an endpoint.")
parser.add_argument("--embeddings_mode", type=bool, default=True, help="Generate embeddings for files instead of inference.")
args = parser.parse_args()

def embed(audio_path, model_name='fc'):
    feature_extractor = AvesEcho(model_name=model_name, slist=args.slist, flist=args.flist,
                            add_filtering=False, mconf=None,
                            outputd="./outputs/temp", avesecho_mapping='./inputs/list_AvesEcho.csv',
                            maxpool=False, add_csv=False, embeddings=args.embeddings_mode, args=args)

    # Generate and save embeddings for files
    embeddings = feature_extractor.generate_embeddings(audio_path)
    return embeddings

# if __name__ == "__main__":
#     embeddings = embed(audio_path="../audio\single\XC487569 - Purple Sandpiper - Calidris maritima.mp3")
#     print(embeddings.shape)


# import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# e1 = torch.load("../audio/benchmark_sound-of-norway/Test/Eurasian Wren/100_2def17a52e173deb6bad32c908_passt.pt", map_location=torch.device(device))
# e2 = torch.load("../audio/benchmark_sound-of-norway/Test/Eurasian Wren/100_2def17a52e173deb6bad32c908.pt", map_location=torch.device(device))
# print(e1)
# print(e2)
# print(e1 == e2)

