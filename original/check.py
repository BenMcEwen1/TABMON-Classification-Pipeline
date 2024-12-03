# Check that annotated labels correspond to species list

from util import load_species_list
import pandas as pd

species_list = load_species_list('./inputs/list_en_ml.csv')

df = pd.read_csv('../benchmark_sound-of-norway/annotation_split.csv')

labels = df['new_label'].unique().tolist()

for label in labels:
    if label.replace(" ", "") not in species_list:
        print(label)
