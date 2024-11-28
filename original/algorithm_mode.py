from enum import Enum


# Allow the algorithm to run in directories mode (processing a batch of audio files from the input directory and writing
# the analysis results to a JSON file in he output directory) or endpoint mode (supporting interactive use by starting a
# Flask web server that responds to post requests).
class AlgorithmMode(Enum):
    DIRECTORIES = "directories"
    ENDPOINT = "endpoint"