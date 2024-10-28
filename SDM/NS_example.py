# Example based on the following tutorial by Daniel Furman https://daniel-furman.github.io/Python-species-distribution-modeling/
# Installed the follwoing packages - scikit-learn, pyimpute, rasterio, geopandas

# Example - Joshua Tree Species Distribution

# import folium.raster_layers
import geopandas as gpd
import matplotlib.pyplot as plt
# import mapclassify
# import folium
import shutil
import glob
import rasterio
from rasterio import warp
import numpy as np
import pandas as pd
# from sklearn.model_selection import RandomizedSearchCV
from osgeo import gdal
# import fiona

# import xarray as xr 
# import rioxarray as rio 

# Set plot text sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# Grid Spec for Figure 4
fig4 = plt.figure(constrained_layout=True,figsize=(24,16))
gs = fig4.add_gridspec(3, 6,height_ratios=[1.0,1.0,0.75])

plot_weeks = [15, 17, 19, 21, 23, 25]
titles = ["April 10-16","April 24-30","May 8-14","May 22-28","June 5-11","June 19-25"]

# Add Willow Warbler Plots
index = 0
for week in plot_weeks:
    ww_aSDM = fig4.add_subplot(gs[0, index])
    ww_aSDM.set_title(titles[index])
    ww_aSDM.set_xticks([])
    ww_aSDM.set_yticks([])
    plt.setp(ww_aSDM.spines.values(), color=None)
    #ww_aSDM.axis('off')
    #ww_aSDM.yaxis.set_visible(False)
    #ww_aSDM.xaxis.set_visible(False)
    if index == 0:
        ww_aSDM.set_ylabel("Willow Warbler\nVocalization Probability")
    index += 1
                     
# Add Spotted Flycatcher Plots
index = 0
for week in plot_weeks:
    sf_aSDM = fig4.add_subplot(gs[1, index])
    sf_aSDM.set_xticks([])
    sf_aSDM.set_yticks([])
    plt.setp(sf_aSDM.spines.values(), color=None)
    #sf_aSDM.axis('off')
    if index == 0:
        sf_aSDM.set_ylabel("Spotted Flycatcher\nVocalization Probability")    

    index += 1

# Instantiate migrating species heatmap
migrating_heatmap = fig4.add_subplot(gs[2, 0:2])
migrating_heatmap.set_ylabel("Latitude Band")
migrating_heatmap.set_title('aSDM Richness of Full Migrant Species')
migrating_heatmap.set_xlabel("Date")

# Instantiate violin plots
eBird_vs_aSDM_plot = fig4.add_subplot(gs[2, 2:4])
eBird_vs_aSDM_plot.set_title('aSDM Distributions at eBird Survey Points')
eBird_vs_aSDM_plot.set_ylabel("Vocalization Probability")
eBird_vs_aSDM_plot.set_xlabel("Species")

BBS_vs_aSDM_plot = fig4.add_subplot(gs[2, 4:6])
BBS_vs_aSDM_plot.set_title("aSDM Distributions at BBS Survey Points")
BBS_vs_aSDM_plot.set_ylabel("Vocalization Probability")
BBS_vs_aSDM_plot.set_xlabel("Species")

def convertToTIFFFromNC(filepath: str, attr:str, output_path:str, crs:str="epsg:4326"):
    """
     Convert standardized earth observation NetCDF variables 
     to single-band georeferencing TIFF file (GeoTIFF).
    """
    nc_file = xr.open_dataset(filepath)
    variable = nc_file[attr]
    variable = variable.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    variable.rio.write_crs(crs, inplace=True)
    variable.rio.to_raster(output_path)

# convertToTIFFFromNC(
#     filepath="SDM/covariates/forest_cover/C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.72.780603.-17.586704.35.153385.34.757088.nc", 
#     attr="lccs_class", # Land Cover Classification System (LCCS)
#     output_path="SDM/covariates/forest_cover/forest_cover.tif"
#     )

cwd = "National_PAM_of_Biodiversity_Bick_et_al_2024"
covariates = pd.read_csv(cwd+"/Data/Sites/sites_weekly_covariates_zeroFill_NDVI_forRelease.csv")
weeks = list(range(11,30))
common_resolution = 0.1
output_folder = "example_outputs/"

def generate_covariate_tiffs(weeks, cwd, output_folder, common_resolution):
    for week in weeks:
        # Load covariate tiffs for this week
        input_files = [
                    cwd+'/Data/Covariates/Norwegian_Meteorological_Institute/epsg_4326/weekly/NDVI/{}/NDVI_avg_{}.tif'.format(str(week),str(week)),
                    cwd+'/Data/Covariates/Norwegian_Meteorological_Institute/epsg_4326/weekly/tx/{}/tx_avg_{}.tif'.format(str(week),str(week)),
                    cwd+'/Data/Covariates/Norwegian_Meteorological_Institute/epsg_4326/weekly/tg/{}/tg_avg_{}.tif'.format(str(week),str(week)),
                    cwd+'/Data/Covariates/Norwegian_Meteorological_Institute/epsg_4326/weekly/tn/{}/tn_avg_{}.tif'.format(str(week),str(week)),
                    cwd+'/Data/Covariates/Norwegian_Meteorological_Institute/epsg_4326/weekly/rr/{}/rr_avg_{}.tif'.format(str(week),str(week)),
                    cwd+'/Data/Covariates/Altitude/DTM10_UTM33_merged_1000_bilinear_4326.tif'
                    ]

        print(input_files)

        # Read single-band raster files
        raster_datasets = [rasterio.open(file) for file in input_files]
        print(raster_datasets)
        
        # Find common resolution (minimum cell size)
        #common_resolution = min(ds.res[0] for ds in raster_datasets)

        # Find common extent (maximum bounding box)
        min_x = min(ds.bounds.left for ds in raster_datasets)
        min_y = min(ds.bounds.bottom for ds in raster_datasets)
        max_x = max(ds.bounds.right for ds in raster_datasets)
        max_y = max(ds.bounds.top for ds in raster_datasets)
        common_extent = (min_x, min_y, max_x, max_y)

        # Define the shape of the output grid in the target CRS
        dst_shape = (
            int((common_extent[3] - common_extent[1]) / common_resolution),
            int((common_extent[2] - common_extent[0]) / common_resolution)
        )

        # Create an empty array for the resampled data
        multiband_array = np.empty((dst_shape[0], dst_shape[1], len(raster_datasets)), dtype=rasterio.float32)
        # print(multiband_array.shape)

        raster_list = []
        # Read and resample each raster to the common extent and resolution
        for i, ds in enumerate(raster_datasets):
            # Reproject the source data to the destination grid
            resampled_raster, _ = rasterio.warp.reproject(
                source=ds.read(1),
                destination=multiband_array[:, :, i],
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=rasterio.transform.from_origin(common_extent[0], common_extent[3], common_resolution, common_resolution),
                dst_crs=ds.crs,
                resampling=rasterio.warp.Resampling.nearest  # Choose a resampling method
            )
            raster_list.append(resampled_raster)

        # Stack the data along the third axis to create a multiband array
        multiband_array = np.stack(raster_list, axis=-1)

        # Specify the output file path (replace 'output.tif' with your desired file name)
        output_path = output_folder + 'covars_week_{}.tif'.format(str(week))

        metadata = {
            'driver': 'GTiff',
            'count': 6,  # Replace with the number of bands in your multiband array
            'dtype': 'float32',  # Replace with the data type of your multiband array
            'width': multiband_array.shape[1],  # Replace with the width of your multiband array
            'height': multiband_array.shape[0],  # Replace with the height of your multiband array
            'crs': 'EPSG:4326',  # Replace with the CRS of your data
            'transform': rasterio.transform.from_origin(common_extent[0], common_extent[3], common_resolution, common_resolution)
        }

        # Write the multiband array to a new raster file
        num_bands = multiband_array.shape[-1]

        # print(multiband_array.shape)
        multiband_array = multiband_array[0]
        multiband_array = np.transpose(multiband_array, (2, 0, 1))
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(multiband_array)

generate_covariate_tiffs(weeks, cwd, common_resolution=common_resolution, output_folder=output_folder)


def add_covariates(audio_presence,covars,week):
    for covar in covars:
        audio_presence[covar] = np.nan
        for week in weeks:
            field_name = covar + '_' + str(week)
            covariates_temp = covariates[["site",field_name]]
            for i, row in audio_presence.iterrows():
                if row["week"]==week:
                    row_site = row["site"]
                    val = covariates_temp.loc[covariates_temp['site'] == row_site, field_name].iloc[0]
                    audio_presence.at[i,covar] = val

    # Also add the altitude values
    covars = covars + ["altitude"]

    # drop nan weeks without NDVI values
    audio_presence = audio_presence[audio_presence['NDVI'].notna()]

    return audio_presence

def runRFParameterSearch(model,n_iter=100):
    # Create Randomizied Hyperparameter Search, Code from: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 200)]
    # Number of features to consider at every split
    max_features = [None, 'sqrt','log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]    

    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    
    # Use the random grid to search for best hyperparameters

    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = model, 
                                   param_distributions = random_grid, 
                                   n_iter = n_iter, 
                                   scoring='neg_mean_squared_error',
                                   cv = 3, 
                                   verbose=0, 
                                   random_state=21, 
                                   n_jobs = -1)# Fit the random search model

    return rf_random

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn import metrics

def trainRandomForestAudioSDM(X_audio,y_audio,X_train,y_train,X_test,y_test,covars,sp,n_iter):
    # Count the number of detections (1) and non-detections (0)
    count_0 = y_train.to_list().count(0)
    count_1 = y_train.to_list().count(1)

    # Calculate class fractions to weight random search models
    if count_0 > count_1:
        weight_1 = count_0/count_1
        print('weight 1 to 0: ', weight_1)
        sample_weight = np.array([weight_1 if i == 1 else 1 for i in y_train])

    if count_1 > count_0:
        weight_0 = count_1/count_0
        print('weight 1 to 0: ', weight_0)
        sample_weight = np.array([weight_0 if i == 0 else 1 for i in y_train])


    # Initialize Model
    model = RandomForestRegressor()

    # Run random parameter search
    rf_random = runRFParameterSearch(model,n_iter)
    # Fit random search model
    rf_random.fit(X_train, y_train,sample_weight=sample_weight)

    print('best params')
    print(rf_random.best_params_)

    # Generate a base model for results comparison
    base_model = RandomForestRegressor(n_estimators = 1000, random_state = 42)

    # Fit base model and calculate error
    base_model.fit(X_train, y_train)
    base_train_predictions = base_model.predict(X_train)
    base_test_predictions = base_model.predict(X_test)
    base_train_mae = mean_absolute_error(y_train, base_train_predictions)
    base_test_mae = mean_absolute_error(y_test, base_test_predictions)

    # Get best parameter-searched models based on test scores
    best_random = rf_random.best_estimator_

    # Calculate aSDM predictions based on covariates 
    random_train_predictions = best_random.predict(X_train)
    random_test_predictions = best_random.predict(X_test)

    # Calculate error
    random_train_mae = mean_absolute_error(y_train, random_train_predictions)
    random_test_mae = mean_absolute_error(y_test, random_test_predictions)

    random_train_mse = root_mean_squared_error(y_train, random_train_predictions)
    random_test_mse = root_mean_squared_error(y_test, random_test_predictions)

    random_train_r2 =  metrics.r2_score(y_train, random_train_predictions)
    random_test_r2 =  metrics.r2_score(y_test, random_test_predictions)

    # Save errors to dict
    error_dict ={
        'species':sp,
        'random_train_mae':random_train_mae,
        'random_test_mae':random_test_mae,
        'random_train_mse':random_train_mse,
        'random_test_mse':random_test_mse,
        'Best Parameters': rf_random.best_params_,
        'Best Score': rf_random.best_score_
    }

    #print('base train MAE: ', base_train_mae)
    #print('base test MAE: ', base_test_mae)
    #print('random train MAE: ', random_train_mae)
    #print('random test MAE: ', random_test_mae)
    #print('random train R^2:', metrics.r2_score(y_train, random_train_predictions))
    #print('random test R^2:', metrics.r2_score(y_test, random_test_predictions))

    # Get and plot feature importances
    model = best_random

    # Random Forest Importance
    feature_names = covars
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Random Forest Importance - {}".format(sp))
    ax.set_ylabel("Mean Decrease in Impurity")
    fig.tight_layout()

    # Permutation Importance
    result = permutation_importance(
        model, X_test, y_test, n_repeats=100, random_state=42, n_jobs=2
    )

    permutation_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Permutation Importance - {}".format(sp))
    ax.set_ylabel("Mean Accuracy Decrease")
    fig.tight_layout()
    plt.show()

    return model, permutation_importances,forest_importances, error_dict


from rasterio.warp import calculate_default_transform, reproject, Resampling

# Resample Covariates to Common Resolution
def resample_raster(input_raster_path, reference_raster_path, output_raster_path):
    # Open the input raster
    with rasterio.open(input_raster_path) as src:
        # Open the reference raster to get resolution and projection information
        with rasterio.open(reference_raster_path) as dst:
            # Calculate the new transformation matrix to match the reference raster's resolution
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, dst.crs, src.width, src.height, *src.bounds, resolution=dst.res[0])

            # Define the resampled raster's profile
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst.crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Create the output raster file
            with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
                # Perform the resampling and alignment
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst.crs,
                        resampling=Resampling.bilinear)

# Reproject to Common CRS
def reproject_raster(raster, reference_crs):
    # Reproject the raster to match the CRS of the reference raster
    with rasterio.open(
        'temp_raster.tif', 'w',
        driver='GTiff',
        height=raster.height,
        width=raster.width,
        count=raster.count,
        dtype='float64',
        crs=reference_crs,
        transform=raster.transform
    ) as dst:
        for i in range(1, raster.count + 1):
            reproject(
                source=raster.read(i),
                destination=rasterio.band(dst, i),
                src_transform=raster.transform,
                src_crs=raster.crs,
                dst_transform=raster.transform,
                dst_crs=reference_crs,
                resampling=Resampling.nearest
            )

    return rasterio.open('temp_raster.tif')

import sys

# Clip Raster to Norway Extent
def clip_raster(raster, extent):
    # Get the window for clipping
    window = raster.window(*extent)

    # Read the raster data within the specified window
    clipped_data = raster.read(1, window=window)

    # Update the raster profile with the new window and transform
    profile = raster.profile
    profile.update({
        'height': window.height,
        'width': window.width,
        'transform': raster.window_transform(window)
    })

    return clipped_data, profile

# Set NaN where there is less than majority forest cover
def set_nan_rasters(raster1_path, raster2_path, output_path):
    # Open the second raster to get its CRS
    with rasterio.open(raster2_path) as forest:

        # Open and reproject the first raster to match the CRS of the second raster
        with rasterio.open(raster1_path) as aSDM:

            if forest.crs != aSDM.crs:
                print('crs not equal')
                sys.exit()

            # Read as numpy arrays
            aSDM_band = aSDM.read(1)
            forest_band = forest.read(1)

            # Set to NaN where less than 0.5
            aSDM_band[forest_band <= 0.5] = np.nan

            # Write the result to a new raster file
            profile = aSDM.profile
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(aSDM_band, 1)

# Align rasters to same grid
def warp_align_raster(resampled_forest_path, align_forest_path, output_tiff_path):
    # Open the resampled forest raster
    resampled_ds = gdal.Open(resampled_forest_path)
    if resampled_ds is None:
        print(f"Error: Could not open {resampled_forest_path}")
        return

    # Get the dimensions and geotransform of the reference raster
    reference_ds = gdal.Open(output_tiff_path)
    if reference_ds is None:
        print(f"Error: Could not open {output_tiff_path}")
        return
    reference_proj = reference_ds.GetProjection()
    reference_geotrans = reference_ds.GetGeoTransform()
    x_res = reference_geotrans[1]
    y_res = -reference_geotrans[5]  # Negative because y-coordinates usually decrease from top to bottom
    
    # Create warp options
    warp_options = gdal.WarpOptions(format='GTiff',
                                     xRes=x_res,
                                     yRes=y_res,
                                     outputBounds=[reference_geotrans[0],
                                                   reference_geotrans[3] - resampled_ds.RasterYSize * y_res,
                                                   reference_geotrans[0] + resampled_ds.RasterXSize * x_res,
                                                   reference_geotrans[3]],
                                     #width=output_width,
                                     #height=output_height,
                                     
                                     targetAlignedPixels=True,
                                     dstSRS=reference_proj)

    # Perform the warp
    gdal.Warp(align_forest_path, resampled_ds, options=warp_options)

    # Close datasets
    resampled_ds = None
    reference_ds = None

def vocalization_probability(species,weeks,model_name,model,output_folder,cwd):
    norway_path = cwd+"/Data/Sites/norway-stanford-jm135gj5367-shapefile/jm135gj5367.shp"
    norway = gpd.read_file(norway_path)
    # Set path for forest raster and resampled version
    forest_raster_path = cwd+"/Data/Covariates/Forest_Cover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1_clip_forests 2.tif"
    align_forest_path = cwd+"/Data/Covariates/Forest_Cover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1_clip_forests 2_norway_ALIGN.tif"
    resampled_forest_path = cwd+"/Data/Covariates/Forest_Cover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1_clip_forests 2_norway_RESAMP.tif" 

    index = 0
    for week in weeks:
        print('week {}'.format(str(week)))
        print('Predicting Vocalization Probability Week {}'.format(week))
        # Step 1: Read the Multiband TIFF File
        multiband_tiff_path = output_folder + 'covars_week_{}.tif'.format(str(week))
        with rasterio.open(multiband_tiff_path) as src:
            multiband_image = src.read()

            #multiband_image, transform = rasterio.mask.mask(src, norway, crop=False, nodata=None) # NOTE


        # Step 2: Prepare the Data for Prediction
        num_bands, height, width = multiband_image.shape
        data = multiband_image.reshape(num_bands, -1).T  # Reshape to (num_pixels, num_bands)

        flattened_data = pd.DataFrame({'NDVI': data[:, 0], 
                                        'tx': data[:, 1],
                                        'tg': data[:, 2],                                      
                                        'tn': data[:, 3],
                                        'rr': data[:, 4],
                                        'altitude': data[:, 5]                                                                                                     
                                        })

        # Create metadata for the output raster file
        meta = src.meta
        meta.update(dtype='float32', count=1)  # Assuming single-band output

        # Step 4: Predict Habitat Suitability
        predictions = model.predict(flattened_data)

        # Reshape the predictions back to the original image shape
        flattened_data = flattened_data.to_numpy()
        prediction_image = predictions.reshape((height, width))

        # Write the predicted habitat suitability values to a new TIFF file
        output_tiff_path = output_folder + 'suitability_week_{}_{}_{}_audio.tif'.format(str(week), species.replace(" ",""),model_name)
        plot_tiff_path = output_folder + 'suitability_week_{}_{}_{}_plot_audio.tif'.format(str(week), species.replace(" ",""), model_name)

        with rasterio.open(output_tiff_path, 'w', **meta) as dst:
            dst.write(prediction_image, 1)  # Write to band 1

        with rasterio.open(output_tiff_path) as src:
            print(norway.geometry)
            print(src.bounds)
            out_image, transformed = rasterio.mask.mask(src, norway.geometry, crop=True, nodata=np.nan)
            out_profile = src.profile.copy()

        out_profile.update({'width': out_image.shape[2],'height': out_image.shape[1], 'transform': transformed})
        with rasterio.open(output_tiff_path, 'w', **out_profile) as dst:
            dst.write(out_image)



        # Resample forest cover raster
        if index==0:
            print('resample forest cover on index 0')
            # Reproject forest to norway bounds

            # Crop forest to norway bounds
            with rasterio.open(forest_raster_path) as src:
                out_image, transformed = rasterio.mask.mask(src, norway.geometry, crop=True, filled=True)
                out_profile = src.profile.copy()
            
            
            out_profile.update({'width': out_image.shape[2],'height': out_image.shape[1], 'transform': transformed})
            with rasterio.open(resampled_forest_path, 'w', **out_profile) as dst:
                dst.write(out_image)

        
            # Resample and align the input raster
            resample_raster(forest_raster_path, output_tiff_path, resampled_forest_path)
            warp_align_raster(resampled_forest_path, align_forest_path, output_tiff_path) # NOTE

            # Get dims of rasters
            forest = rasterio.open(align_forest_path)
            forest_width = forest.width
            forest_height = forest.height

            aSDM_tiff = rasterio.open(output_tiff_path)
            aSDM_width = aSDM_tiff.width
            aSDM_height = aSDM_tiff.height

            # Correct for warp align
            if forest_width >= aSDM_width and forest_height>=aSDM_height:

                correction_cols = forest_width - aSDM_width
                correction_rows = forest_height - aSDM_height         

                # Drop extra column given by gdal in warp align
                with rasterio.open(align_forest_path) as src:
                    # Read raster data
                    data = src.read()

                    # Drop the rightmost column
                    data_without_extra = data[:, :,-correction_rows :-correction_cols]

                    # Get metadata for the new raster
                    profile = src.profile

                    # Update the metadata to reflect the change in raster dimensions
                    profile['width'] -= correction_cols
                    profile['height'] -= correction_cols

                    # Write the modified array to a new raster file
                    with rasterio.open(align_forest_path, 'w', **profile) as dst:
                        dst.write(data)        
            else:
                print('warp align dim error, try different resolution')
                sys.exit()

        # Multiply aSDM by forest cover mask to remove values in non-forest areas
        set_nan_rasters(output_tiff_path,align_forest_path,plot_tiff_path)

        index += 1

from sklearn.model_selection import train_test_split
model_name = "RF"
hyper_iter = 1

def plot_aSDM(species,weeks,fig4,plot_dict,cwd,labels,row):
    # Source and target CRS (EPSG codes) for aSDMs
    original_epsg = 4326 
    target_epsg = 32632  

    # Load Shapefiles
    norway = gpd.read_file(cwd+"/Data/Sites/norway-stanford-jm135gj5367-shapefile/jm135gj5367.shp")
    norway = norway.to_crs("epsg:{}".format(target_epsg))

    # Define colormap and colorbar properties
    cmap = 'viridis'
    vmin = 0
    vmax = 1.0

    # Iterate over weeks and add clipped images to list
    images = []
    axes = []

    index = 0
    for week in weeks:
        print('week {} index {}'.format(week, index))
        output_tiff_path = output_folder + 'suitability_week_{}_{}_{}_plot_audio.tif'.format(str(week), species.replace(" ",""), model_name)
        reprojected_tiff_path = output_folder + 'suitability_week_{}_{}_{}_epsg{}_audio.tif'.format(str(week), species.replace(" ",""), model_name, str(target_epsg))

        # Load raster and reproject       
        with rasterio.open(output_tiff_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_epsg, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_epsg,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(reprojected_tiff_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_epsg,
                        resampling=Resampling.mode)

        # Crop forest to norway bounds
        with rasterio.open(reprojected_tiff_path) as src:
            print(src.meta)
            src.meta['nodata'] = np.nan

            out_image, transformed = rasterio.mask.mask(src, norway.geometry, crop=True, filled=True,nodata=np.nan)
            out_profile = src.profile.copy()
        
        
        out_profile.update({'width': out_image.shape[2],'height': out_image.shape[1], 'transform': transformed})
        with rasterio.open(reprojected_tiff_path, 'w', **out_profile) as dst:
            dst.write(out_image)

        # Get mean of each aSDM map and plot
        with rasterio.open(reprojected_tiff_path) as src:
            # Read the raster data as a numpy array
            raster_data = src.read(1, masked=True)  # Assuming a single band raster

            # Mask out NaN values and take mean
            masked_data = np.ma.masked_invalid(raster_data)
            mean_value = masked_data.mean()

        # Get the extent from the shapefile
        shapefile_extent = norway.bounds
        shapefile_extent = [shapefile_extent.minx.min(), shapefile_extent.maxx.max(),
                            shapefile_extent.miny.min(), shapefile_extent.maxy.max()]
        

        # Get the correct column index based on the week
        plot_index = plot_dict[week]

        # Add subplot to the specified location in the gridspec
        ax = fig4.add_subplot(gs[row, plot_index])

        if row == 0 and index == 0:
            ax.set_ylabel("Willow Warbler")
        if row == 1 and index == 0:
            ax.set_ylabel("Spotted Flycatcher")
        print('plot')

        norway.plot(ax=ax, edgecolor='grey', facecolor="grey",linewidth=0.0,alpha=0.5,zorder=0)
        ax.annotate(labels[index], (0,0.95), fontsize=25, va='top', ha='left', xycoords='axes fraction')
        ax.annotate("mean:\n{:#.3g}".format(mean_value), (0,0.85), fontsize=16, va='top', ha='left', xycoords='axes fraction')

        with rasterio.open(reprojected_tiff_path) as aSDM:
            aSDM_plot = rasterio.plot.show(aSDM.read(1), transform=aSDM.transform, ax=ax, cmap='YlOrRd',alpha=1.0,zorder=10,vmin=0,vmax=1.0) #,extent=shapefile_extent,)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
  
        if row == 0 and index == 5:
            im = aSDM_plot.get_images()[0]
            fig4.colorbar(im, ax=ax,label='Probability of Vocalization')

        index +=1


# if __name__ == "__main__":
# Store Model Metrics in Lists
df_metrics_lists = []

full_migrant_species  = ["Barn Swallow", "Common Chiffchaff", "Common Cuckoo", "Common Sandpiper", "Eurasian Blackcap", "Green Sandpiper", "White Wagtail", "Willow Warbler", "Spotted Flycatcher", "European Pied Flycatcher"]
# Iterate over species and train models
for sp in full_migrant_species:
    # Load the Audio Detection Data into a geodataframe
    print('{}'.format(sp))
    df_path = cwd+"/Data/Detections/weekly/"
    df = pd.read_csv(df_path + "{}_weekly_audio_detections.csv".format(sp))
    audio_presence= gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    print('loaded audio detections for {}'.format(sp))

    # Add altitude variable
    audio_presence = pd.merge(audio_presence, covariates[["altitude","site"]],on="site")

    # Add other covariates to the dataframe
    covars = ["NDVI","tx","tg","tn","rr"]
    audio_presence = add_covariates(audio_presence,covars,weeks)
    print('added covariates for {}'.format(sp))

    # Add altitude to covariates
    covars = covars + ["altitude"]

    # Get min and max altitudes of recorders
    max_altitude_recorders = audio_presence["altitude"].max()
    min_altitude_recorders = audio_presence["altitude"].min()

    # Set vars and targets
    X_audio = audio_presence[covars]  # Add climate covariates
    y_audio = audio_presence['detected']  # Add your species presence/absence column

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_audio, y_audio, test_size=0.3, shuffle=True, stratify=y_audio, random_state=42)

    print('running aSDM Training for {}'.format(sp))    

    # Train Random Forest aSDM with HyperParameter Search, Set n_iter for Hyperparameter Grid Search
    model, permutation_importances,forest_importances, error_dict = trainRandomForestAudioSDM(X_audio,y_audio,X_train,y_train,X_test,y_test,covars,sp,n_iter=hyper_iter)

    print('Generating aSDM Plots for {}'.format(sp))   

    # Apply aSDM model to covariates and save output 
    vocalization_probability(sp,weeks,model_name,model,output_folder,cwd)

    # Add row to metrics DF
    list_metrics = [error_dict['species'],error_dict['random_train_mae'],error_dict['random_test_mae'],error_dict['random_train_mse'],error_dict['random_test_mse'],error_dict['Best Parameters']]
    df_metrics_lists.append(list_metrics)

# Save metrics
cols = ['Species', 'Train MAE', 'Test MAE', 'Train MSE', 'Test MSE', 'Best Parameters']
df_metrics = pd.DataFrame(df_metrics_lists, columns=cols)
df_metrics.to_csv(cwd+"/Results/aSDM_Model_Metrics.csv")
