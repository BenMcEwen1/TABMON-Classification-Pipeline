import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Database:
    def __init__(self):
        self.vectors = torch.load("embeddings.pt", map_location=torch.device(device))

    def __len__(self):
        # return number of embeddings
        return 
    
    def generate_embeddings():
        # Transfer embedding code: check if embeddings are the same
        pass

    def save():
        pass

    def search():
        # Load Faiss and perform similarity search
        pass