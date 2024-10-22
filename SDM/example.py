# Example based on the following tutorial by Daniel Furman https://daniel-furman.github.io/Python-species-distribution-modeling/
# Installed the follwoing packages - scikit-learn, pyimpute, rasterio, geopandas

# Example - Joshua Tree Species Distribution

import geopandas as gpd
import matplotlib.pyplot as plt
import shutil
import glob

# Load data
for f in sorted(glob.glob('SDM/data/jtree*')):
    shutil.copy(f,'SDM/inputs/')

pa = gpd.GeoDataFrame.from_file("SDM/inputs/jtree.shp")


# Check data for duplicates or NaN
print("number of duplicates: ", pa.duplicated(subset='geometry', keep='first').sum())
print("number of NA's: ", pa['geometry'].isna().sum())
print("Coordinate reference system is: {}".format(pa.crs))
print("{} observations with {} columns".format(*pa.shape))

# Plot species presence
pa[pa.CLASS == 1].plot(marker='*', color='green', markersize=8)

# Plot background points
pa[pa.CLASS == 0].plot(marker='+', color='black', markersize=4)


## Mapping Species Suitability
# grab climate features - cropped to joshua tree study area
for f in sorted(glob.glob('SDM/data/bioclim/bclim*.asc')):
    shutil.copy(f,'SDM/inputs/')
raster_features = sorted(glob.glob(
    'SDM/inputs/bclim*.asc'))
# check number of features 
print('\nThere are', len(raster_features), 'raster features.')

from pyimpute import load_training_vector
from pyimpute import load_targets
train_xs, train_y = load_training_vector(pa, raster_features, response_field='CLASS')
target_xs, raster_info = load_targets(raster_features)
print(train_xs.shape, train_y.shape) # check shape, does it match the size above of the observations?


# import machine learning classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

CLASS_MAP = {
    'rf': (RandomForestClassifier()),
    'et': (ExtraTreesClassifier())
    }

from pyimpute import impute
from sklearn import model_selection
# model fitting and spatial range prediction
for name, (model) in CLASS_MAP.items():
    # cross validation for accuracy scores (displayed as a percentage)
    k = 5 # k-fold
    kf = model_selection.KFold(n_splits=k)
    accuracy_scores = model_selection.cross_val_score(model, train_xs, train_y, cv=kf, scoring='accuracy')
    print(name + " %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, accuracy_scores.mean() * 100, accuracy_scores.std() * 200))
    
    # spatial prediction
    model.fit(train_xs, train_y)
    impute(target_xs, model, raster_info, outdir='SDM/outputs/' + name + '-images',
           class_prob=True, certainty=True)
    
from pylab import plt
# define spatial plotter
def plotit(x, title, cmap="Blues"):
    plt.imshow(x, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title(title, fontweight = 'bold')
    plt.show()

import rasterio
distr_rf = rasterio.open("SDM/outputs/rf-images/probability_1.0.tif").read(1)
distr_et = rasterio.open("SDM/outputs/et-images/probability_1.0.tif").read(1)
distr_averaged = (distr_rf + distr_et)/2

plotit(distr_averaged, "Joshua Tree Range, averaged", cmap="Greens")