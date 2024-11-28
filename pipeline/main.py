from models.avesecho import AvesEcho

# TODO
# I have a csv containing all annotations with (filename, species)
# Given a directory which contains training and test samples I need to:
# - Precompute the BirdNet embeddings for all sub-directories /embedding/1_50a4b521a1375a8b12b85643c1_{split_n}.np
# - Training and evaluation loops for data

# Other:
# - Maybe adjust species list and number of classes 
# - Incorporate location and time information
# - Incorporate in-context embeddings


def run_inference():
    """
    Run the AvesEcho model on a set of audio files.

    Parameters
    ----------
    add_filtering: None or str
        Enable geographic filtering
    mconf: None or str
        If not None, the name of a configuration file for the filtering
        method. This file should contain the parameters for the selected
        filtering method.
    maxpool: bool
        If True, the model will use max pooling to reduce the dimensionality
        of the input data before classification.
    add_csv: bool
        If True, the model will output its results in a CSV file, in addition
        to the JSON file.
    lat: None or float
        The latitude of the audio recording location.
    lon: None or float
        The longitude of the audio recording location.
    model_name : str
        The name of the model to use. Options include "passt" and "fc".
    slist: str
        The path to the species list file.
    flist: str
        The path to the folder containing the audio files to classify.
    avesecho_mapping : str
        The path to the AvesEcho mapping file.
    outputd: str
        The path to the output directory for the algorithm.
    result_file: str
        The path to the JSON file containing the results of the algorithm.

    Returns: None
    -------
    """

    add_filtering = None
    mconf = None
    maxpool = False
    add_csv = True
    lat = None
    lon = None

    model_name = "fc"
    slist = "./pipeline/inputs/list_sp_ml.csv"
    flist = "./audio/SoN_single"
    avesecho_mapping = "./pipeline/inputs/list_AvesEcho.csv"

    outputd = "./pipeline/inputs/temp"
    result_file = './pipeline/outputs/analysis-results.json'

    classifier = AvesEcho(model_name=model_name, slist=slist, flist=flist,
                          add_filtering=add_filtering, mconf=mconf,
                          outputd=outputd, avesecho_mapping=avesecho_mapping,
                          maxpool=maxpool, add_csv=add_csv)

    classifier.analyze(audio_input="./audio/SoN_single", lat=lat, lon=lon, result_file=result_file)


if __name__ == "__main__":
    run_inference()