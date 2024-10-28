"""
Example for plotting a raster image on a map using Folium
Output - map_raster.html
Issue - Current alignment issue
"""

import folium
import rasterio
from rasterio import warp
import geopandas as gpd
import rasterio.mask
import json
import numpy as np
import matplotlib.pyplot as plt

# src = rasterio.open(r"SDM\europe_elevation_map.tif")
src = rasterio.open(r"SDM\National_PAM_of_Biodiversity_Bick_et_al_2024\Data\Covariates\Altitude\DTM10_UTM33_merged_1000_bilinear_4326.tif")
rasdata = src.read(1)

# plt.imshow(rasdata)
# plt.show()

left, bottom, right, top = [i for i in src.bounds]

# import fiona
# with fiona.open("tests/data/box.shp", "r") as shapefile:
#     shapes = [feature["geometry"] for feature in shapefile]

# Crop raster to extent of Norway
norway = gpd.GeoDataFrame.from_file("SDM/National_PAM_of_Biodiversity_Bick_et_al_2024/Data/Sites/norway-stanford-jm135gj5367-shapefile/jm135gj5367.shp")

## Convert to GeoJSON (one-off)
# esri_shapefile = gpd.read_file(r'input_esri_shapefile')
# export_geojson = norway.to_file(r'output_export_geojson', driver='GeoJSON')


geojson = gpd.read_file(r'output_export_geojson')

def getFeatures(gdf):
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

coords = getFeatures(geojson)
cropped, transform = rasterio.mask.mask(dataset=src, shapes=coords, crop=True, nodata=0)
print(cropped.shape)

cropped = np.reshape(cropped, (cropped.shape[0]*cropped.shape[1], cropped.shape[2]))
print(cropped.shape)

# curr_crs = 'EPSG:9001'
# dest_crs = 'EPSG:4326'
# left, bottom, right, top = warp.transform_bounds(src_crs=curr_crs, dst_crs=dest_crs, left=left, bottom=bottom, right=right, top=top)

map = folium.Map([40.7, -74], zoom_start=1)
bounds = [[bottom, left], [top, right]]

# Attempt using geopandas (Doesn't work)
# for index, row in norway.iterrows():
#     cropped, transform = rasterio.mask.mask(dataset=src, shapes=[row['geometry']][0], crop=True, nodata=0)

# Attempt using fiona
import fiona

shapefile_path = 'SDM/National_PAM_of_Biodiversity_Bick_et_al_2024/Data/Sites/norway-stanford-jm135gj5367-shapefile/jm135gj5367.shp'
with fiona.open(shapefile_path, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

tiffile_path = r'SDM\National_PAM_of_Biodiversity_Bick_et_al_2024\Data\Covariates\Altitude\DTM10_UTM33_merged_1000_bilinear_4326.tif'
with rasterio.open(tiffile_path, "r") as src:
    print(src.read(1))
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, filled=True)
    print(out_image)
    out_meta = src.meta
    left, bottom, right, top = [i for i in src.bounds]
    bounds = [[bottom, left], [top, right]]
    # out_meta.update({"driver": "GTiff",
    #              "height": out_image.shape[1],
    #              "width": out_image.shape[2],
    #              "transform": out_transform})

# with rasterio.open("cropped.tif", "w", **out_meta) as dest:
#     dest.write(out_image)

# src = rasterio.open(tiffile_path)
# src = rasterio.open("cropped.tif")
# cropped = src.read(1)
# left, bottom, right, top = [i for i in src.bounds]
# bounds = [[bottom, left], [top, right]]

folium.raster_layers.ImageOverlay(out_image[0], bounds=bounds, opacity=0.8).add_to(map)

folium.GeoJson(geojson, name="Norway").add_to(map)

folium.LayerControl().add_to(map)

output_file = "./map_raster.html"
map.save(output_file)


# plt.imshow(out_image[0], cmap='pink')
# plt.show()





### Joshua Tree Example

# # Load data
# for f in sorted(glob.glob('SDM/data/jtree*')):
#     shutil.copy(f,'SDM/inputs/')

# # Check data for duplicates or NaN
# print("number of duplicates: ", pa.duplicated(subset='geometry', keep='first').sum())
# print("number of NA's: ", pa['geometry'].isna().sum())
# print("Coordinate reference system is: {}".format(pa.crs))
# print("{} observations with {} columns".format(*pa.shape))

# # Plot species presence
# pa[pa.CLASS == 1].plot(marker='*', color='green', markersize=8)

# # Plot background points
# pa[pa.CLASS == 0].plot(marker='+', color='black', markersize=4)

# # gpd.explore("area", legend=False)

# ## Mapping Species Suitability
# # grab climate features - cropped to joshua tree study area
# for f in sorted(glob.glob('SDM/data/bioclim/bclim*.asc')):
#     shutil.copy(f,'SDM/inputs/')
# raster_features = sorted(glob.glob(
#     'SDM/inputs/bclim*.asc'))
# # check number of features 
# print('\nThere are', len(raster_features), 'raster features.')

# from pyimpute import load_training_vector
# from pyimpute import load_targets
# train_xs, train_y = load_training_vector(pa, raster_features, response_field='CLASS')
# target_xs, raster_info = load_targets(raster_features)
# print(train_xs.shape, train_y.shape) # check shape, does it match the size above of the observations?


# # import machine learning classifiers
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier

# CLASS_MAP = {
#     'rf': (RandomForestClassifier()),
#     'et': (ExtraTreesClassifier())
#     }

# from pyimpute import impute
# from sklearn import model_selection
# # model fitting and spatial range prediction
# for name, (model) in CLASS_MAP.items():
#     # cross validation for accuracy scores (displayed as a percentage)
#     k = 5 # k-fold
#     kf = model_selection.KFold(n_splits=k)
#     accuracy_scores = model_selection.cross_val_score(model, train_xs, train_y, cv=kf, scoring='accuracy')
#     print(name + " %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
#           % (k, accuracy_scores.mean() * 100, accuracy_scores.std() * 200))
    
#     # spatial prediction
#     model.fit(train_xs, train_y)
#     impute(target_xs, model, raster_info, outdir='SDM/outputs/' + name + '-images',
#            class_prob=True, certainty=True)
    
# from pylab import plt
# # define spatial plotter
# def plotit(x, title, cmap="Blues"):
#     plt.imshow(x, cmap=cmap, interpolation='nearest')
#     plt.colorbar()
#     plt.title(title, fontweight = 'bold')
#     plt.show()

# import rasterio
# distr_rf = rasterio.open("SDM/outputs/rf-images/probability_1.0.tif").read(1)
# distr_et = rasterio.open("SDM/outputs/et-images/probability_1.0.tif").read(1)
# distr_averaged = (distr_rf + distr_et)/2

# plotit(distr_averaged, "Joshua Tree Range, averaged", cmap="Greens")