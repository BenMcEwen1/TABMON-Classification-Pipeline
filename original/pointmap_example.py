from embeddings import Embeddings

embedding_path = "../audio/sound-of-norway/Train/embeddings.bin"
embeddings = Embeddings(embeddings_path=embedding_path)

labels = embeddings.collect_labels(label_path="../audio/sound-of-norway/Train/labels.pt",
                                   audio_path="../audio/sound-of-norway/Train/", 
                                   annotations_path="../audio/sound-of-norway/annotation_split.csv")

embeddings.plot_embeddings(method='pca', labels=labels)