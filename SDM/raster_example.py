# Working with raster tiff files

import rasterio
from rasterio.plot import show


raster = rasterio.open(r"SDM\National_PAM_of_Biodiversity_Bick_et_al_2024\Data\Covariates\Altitude\DTM10_UTM33_merged_1000_bilinear_4326.tif")
img = raster.read()

# Raster image bands and dimensions
print(img.shape) # (bands, width, height) in this case (1, 935, 2248) so single band

# Raster CRS
print(f"Coordinate Reference System: {raster.crs}") # EPSG:4326 (WGS84) (raster.crs)

# Raster metadata
print(f"Raster Metadata: {raster.meta}") # (raster.meta)

# Raster plotting
show(img)