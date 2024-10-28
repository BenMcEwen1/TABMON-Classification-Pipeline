import rasterio
import fiona
from rasterio.mask import mask
import folium
import geopandas as gpd

# Intialise Folium map
map = folium.Map([64.5, 17.5], zoom_start=4, crs="EPSG4326")
# NOTE: Interesting when using EPSG4326 alignment issue is resolved but both polygon and raster not positioned correctly

# Load shapefile
shapefile_path = r"SDM\National_PAM_of_Biodiversity_Bick_et_al_2024\Data\Sites\norway-stanford-jm135gj5367-shapefile\jm135gj5367.shp"
with fiona.open(shapefile_path, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
    norway = gpd.GeoDataFrame.from_features([feature for feature in shapefile])
    folium.GeoJson(norway.to_json(), name="Norway").add_to(map)

# Load file
tiffile_path = r"SDM\National_PAM_of_Biodiversity_Bick_et_al_2024\Data\Covariates\Altitude\DTM10_UTM33_merged_1000_bilinear_4326.tif"
with rasterio.open(tiffile_path) as src:
    out_image, out_transform = mask(src, shapes, crop=True)
    out_meta = src.meta
    print(src.crs)

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],  
                 "width": out_image.shape[2]},
                 transform=out_transform)

    with rasterio.open("new.tif", "w", **out_meta) as dest:
        dest.write(out_image)

# Load and crop raster image
with rasterio.open("new.tif") as src:
    cropped = src.read(1)
    left, bottom, right, top = [i for i in src.bounds]
    bounds = [[bottom, left], [top, right]] # [[57.94929969795261, 3.8937728214471816], [71.19182747225568, 31.16322896992494]]

    folium.raster_layers.ImageOverlay(cropped, bounds=bounds, opacity=0.8, ).add_to(map)
    folium.LayerControl().add_to(map)

    output_file = "./new.html"
    map.save(output_file)