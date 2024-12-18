import os
import argparse
import torch
from algorithm_mode import AlgorithmMode
from models import AvesEcho
import faiss
import numpy as np
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

default_algorithm_mode = os.getenv("ALGORITHM_MODE", AlgorithmMode.DIRECTORIES.value)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_name', type=str, default='fc', help='Name of the model to use, either fc or passt')
parser.add_argument('--slist', type=str, default='./inputs/list_sp_ml.csv', help='Path to the species list.')
parser.add_argument('--flist', type=str, default='./inputs/species_list_nl.csv', help='Path to the filter list of species.')
parser.add_argument('--i', type=str, default='audio/NH-11_20240415_062840.WAV', help='Input audio sample.')
parser.add_argument("--algorithm_mode", default=default_algorithm_mode, help="Use input/output directories or an endpoint.")
parser.add_argument("--embeddings_mode", type=bool, default=True, help="Generate embeddings for files instead of inference.")
parser.add_argument("--regenerate", type=bool, default=False, help="If embeddings already exist, regenerate them.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Embeddings:
    def __init__(self, embeddings_path:str=None, dimension:int=320):
        self.dimension = dimension
        self.embeddings_path = embeddings_path

        if embeddings_path is not None:
            self.index = faiss.read_index(self.embeddings_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension) 

    def __len__(self):
        return self.index.ntotal
    
    def generate_embeddings(self, audio_path, model_name:str='fc', regenerate:bool=False, save:bool=True):
        """
        Generate embeddings for the given audio file or directory using the given model.
        Args:
            audio_path (str): The path to the audio file or audio directory.
            model_name (str, optional): The name of the model to use. Defaults to 'fc'.
            regenerate (bool, optional): Whether to regenerate the embeddings even if they already exist. Defaults to False.
        Returns: The generated embedding(s).
        """
        feature_extractor = AvesEcho(model_name=model_name, slist='./inputs/list_sp_ml.csv', flist='./inputs/species_list_nl.csv',
                                    add_filtering=False, mconf=None, outputd="./outputs/temp", avesecho_mapping='./inputs/list_AvesEcho.csv',
                                    maxpool=False, add_csv=False, embeddings=args.embeddings_mode, args=args)
        
        embeddings = feature_extractor.generate_embeddings(audio_path, regenerate, save)
        return embeddings

    def collect(self, audio_path, suffix:str="", index:str='embeddings.bin'):
        # Collect individual embeddings into a database
        for path, _, files in os.walk(audio_path):
            for filename in files:
                embedding_filename = os.path.splitext(filename)[0] + suffix + '.pt'
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension == '.mp3':
                    embedding_path = os.path.join(path, embedding_filename)
                    embedding = torch.load(embedding_path, map_location=torch.device(device))
                    embedding = torch.squeeze(embedding, 1)
                    if embedding.shape[1] == self.dimension:
                        self.index.add(embedding)
        faiss.write_index(self.index, os.path.join(audio_path, index))

    def collect_labels(self, label_path=None, audio_path=None, annotations_path=None):
        if label_path == None:
            annotations = pd.read_csv(annotations_path)
            species = annotations['new_label'].unique().tolist()
            labels = []

            # Iterate through each audio file
            for path, _, files in os.walk(audio_path):
                for filename in files:
                    embedding_path = os.path.splitext(filename)[0] + '.pt'
                    file_extension = os.path.splitext(filename)[1].lower()
                    if file_extension == '.mp3':
                        label = annotations.loc[annotations['Filename'] == filename, 'new_label'].values[0]
                        encoding = species.index(label)
                        embedding = torch.load(os.path.join(path, embedding_path), map_location=torch.device(device))
                        labels.extend([encoding]*embedding.shape[0])
            torch.save(torch.tensor(labels, dtype=torch.float32), os.path.join(audio_path, 'labels.pt'))
        else: 
            labels = torch.load(label_path, map_location=torch.device(device))
        return labels

    def search_embeddings(self, query:torch.Tensor, k:int=2):
        """
        Search for k nearest neighbors of the given query embeddings in the database.
        Args:
            query (torch.Tensor): The query embeddings to search for, input shape (B, 1, embedding_dim).
            k (int, optional): The number of nearest neighbors to return. Defaults to 2.
        
        Returns:
            torch.Tensor: The k nearest neighbors of the query embeddings, with shape (B, k, embedding_dim).
        """
        batch = query.shape[0]
        distances, indices = self.index.search(np.squeeze(query,1), k)

        vectors = self.index.reconstruct_batch(indices.flatten())
        vectors = vectors.reshape(batch, k, 320)
        return vectors, distances
    
    def plot_embeddings(self, method:str='pca', labels=None):
        embeddings = self.index.reconstruct_n(0, self.index.ntotal)

        if method == 'pca':
            pca = PCA(n_components=2)
            points = pca.fit_transform(embeddings)
        elif method == 'umap':
            dim_reduction = umap.UMAP(n_components=2)
            points = dim_reduction.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], c=labels, s=10, alpha=0.7)
        plt.title("2D PCA Projection of embeddings")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()
        

if __name__ == "__main__":
    database = Embeddings()
    database.generate_embeddings(audio_path=args.i, model_name=args.model_name, regenerate=args.regenerate)