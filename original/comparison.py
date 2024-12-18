from embeddings import Embeddings

if __name__ == "__main__":
    audio_path = "../audio/subset/"
    embeddings = Embeddings(dimension=320)
    embeddings.generate_embeddings(audio_path=audio_path, model_name="fc", regenerate=False)

    embeddings.collect(audio_path, suffix='', index='BirdNet_subset.bin')

    labels = embeddings.collect_labels(label_path="../audio/subset/labels.pt",
                                       audio_path=audio_path, 
                                       annotations_path="../audio/sound-of-norway/annotation_split.csv")

    embeddings.plot_embeddings(method='umap', labels=labels)