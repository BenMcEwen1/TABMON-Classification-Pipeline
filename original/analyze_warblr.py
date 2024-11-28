# -*- coding: utf-8 -*-
# @Time    : 19/03/24 14:30 
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl
# @File    : analyze_warblr.py


from models import *
from config import *

       
if __name__ == "__main__":
    """
    # Initialize the AvesEcho classifier with various arguments
     - model_name: Set the model type, 'fc' for CNN or 'passt' for Transformer-based model. Default is 'fc'.
     - slist: Path to the species list file in CSV format.
     - flist: Path to a file listing species for location-based filtering. Required if add_filtering is True.
     - add_filtering: Enables location based filtering employing the `flist`.
     - mconf: The minimum confidence threshold for the model predictions. Defaults to pre-computed species-wise thresholds,
              set to 0 for no threshold.
     - outputd: The directory where temporary audio chunks will be saved.
     - avesecho_mapping: Path to the AvesEcho mapping file.
     - maxpool: If True, the model will generate temporally-summarised output (not relevant for warblr).
     - add_csv: If True, enables output in CSV format in addition to the default JSON output (not relevant for warblr).
   
     - audio_file_path: Path to the audio file. 
     - lat: Latitude for geographic filtering. When set, ignores the `flist`.
     - lon: Longitude for geographic filtering. When set, ignores the `flist`.
    """

    classifier = AvesEcho(model_name='fc', slist='inputs/list_sp_ml.csv', flist='inputs/species_list.csv',
                    add_filtering=False, mconf=None,
                    outputd = 'tmp/avesecho', avesecho_mapping='inputs/list_AvesEcho.csv',
                    maxpool=False, add_csv=False, args=None)
    
    species, probabilities = classifier.analyze_warblr_audio(audio_file_path='audio/20240307_090321.WAV', lat=None, lon=None)
    
    print(f'Predictions for {species} are: {probabilities}')
