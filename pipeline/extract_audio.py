# This file is used to locate the audio segment within the database given an embedding index 
# This is used for checking which segments were sampled

from embeddings import Embeddings
import torch
import os


embedding_path = "../audio/subset/Common Chaffinch/12_63f239ae9435afc2d1ea2d7ee8.pt"

embedding = torch.load(embedding_path, map_location='cpu')

# Collection context
context = Embeddings('../audio/subset/fc_subset.bin', dimension=320)

# Search context
samples, distance = context.search_embeddings(embedding, k=5)
# print(samples.shape)

audio_path = "../audio/subset/"

files_list = []
paths = []
i = 0
for path,_,files in os.walk(audio_path):
    if i > 0:
        files_list += files
        paths += [path]*len(files)
    i += 1

current_index = 0
target_index = 819
for i, filename in enumerate(files_list):
    embedding_filename = os.path.splitext(filename)[0] + '.pt'
    file_extension = os.path.splitext(filename)[1].lower()
    if filename[-5:] == 'fc.pt':
        embedding = torch.load(os.path.join(paths[i], embedding_filename), map_location='cpu')
        length = embedding.shape[0]
        current_index += length
        if current_index > target_index:
            print(f"{filename} at index {current_index-target_index}, starting at {3*(current_index-target_index)} seconds")
            break