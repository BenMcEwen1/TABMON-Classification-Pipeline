from pipeline.embeddings import Embeddings


if __name__ == "__main__":
    audio_path = "../audio/brambling/"
    model_name = "fc"

    if model_name == "fc":
        dimension = 320
    if model_name == "passt":
        dimension = 768
    if model_name == "birdnet":
        dimension = 1024

    embeddings = Embeddings(dimension=dimension)
    
    embeddings.generate_embeddings(audio_path=audio_path, model_name=model_name, regenerate=True)

    embeddings.collect(audio_path, suffix=f'_{model_name}', index='embeddings.bin')

    labels = embeddings.collect_labels(label_path=None,
                                       audio_path=audio_path, 
                                       suffix=f'_{model_name}',
                                       annotations_path="../audio/sound-of-norway_old/annotation_split.csv")

    embeddings.plot_embeddings(method='umap', labels=labels)