from models.avesecho import AvesEcho

def run_algorithm():
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
    add_csv = False
    lat = None
    lon = None

    model_name = "fc"
    slist = "./pipeline/inputs/list_sp_ml.csv"
    flist = "./audio/single"
    avesecho_mapping = "./pipeline/inputs/list_AvesEcho.csv"

    outputd = "./pipeline/inputs/temp"
    result_file = './pipeline/outputs/analysis-results.json'

    classifier = AvesEcho(model_name=model_name, slist=slist, flist=flist,
                          add_filtering=add_filtering, mconf=mconf,
                          outputd=outputd, avesecho_mapping=avesecho_mapping,
                          maxpool=maxpool, add_csv=add_csv)

    classifier.analyze(audio_input="./audio/single", lat=lat, lon=lon, result_file=result_file)


if __name__ == "__main__":
    run_algorithm()