from embeddings import Embeddings

if __name__ == "__main__":
    audio_path = "../audio/sound-of-norway_old/sound-of-norway/Train/"
    embeddings = Embeddings(dimension=585)
    # embeddings.generate_embeddings(audio_path=audio_path, model_name="fc", regenerate=False)

    embeddings.collect(audio_path, suffix='_passt', index='passt_subset.bin')

    labels = embeddings.collect_labels(label_path=None,
                                       audio_path=audio_path, 
                                       annotations_path="../audio/sound-of-norway_old/annotation_split.csv")

    embeddings.plot_embeddings(method='pca', labels=labels)