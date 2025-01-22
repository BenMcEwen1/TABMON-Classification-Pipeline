# -*- coding: utf-8 -*-
# @Time    : 19/10/23 15:30 
# @Author  : Burooj Ghani and Dan Stowell
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl
# @File    : species_list.py


import pandas as pd
import mgrs


def get_species_list(recording_lat, recording_long):
    "Concert WGS84 lat-lon to MGRS UTM coordinates"

    # Load occurances data from csv file
    file_path = "inputs/ebba2_data_occurrence_50km.csv"
    df = pd.read_csv(file_path, delimiter=';')
    
    m = mgrs.MGRS()
    c = m.toMGRS(recording_lat, recording_long)

    input_code = c[0:5] 

    # Filter the DataFrame based on the input code
    filtered_df = df[df['cell50x50'].str.startswith(input_code)]

    # Get the list of unique bird species from the filtered DataFrame
    species_list = pd.DataFrame(filtered_df['birdlife_scientific_name'].drop_duplicates())
    #species_list.loc[len(species_list), 'birdlife_scientific_name'] = 'Noise'
    #species_list.to_csv('species_list.csv', index=False, header=False)
    
    return species_list



##########################
if __name__ == '__main__':
    # For demonstration purposes
    from sys import argv

    if len(argv)==3:
        recording_lat = float(argv[1])
        recording_long = float(argv[2])
    else:
        recording_lat = 38.08  # Example latitude
        recording_long = 24.4  # Example longitude
    print(f"# Species list for {recording_lat}, {recording_long}:")

    species_present = get_species_list(recording_lat, recording_long)

    for sp in sorted(list(species_present['birdlife_scientific_name'])):
        print(sp)

    species_present.to_csv('inputs/species_list_greece.csv', index=False, header=False)


