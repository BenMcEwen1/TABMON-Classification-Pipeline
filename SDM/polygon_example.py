"""
Example of plotting a polygon on a map using Geopandas
"""

import geopandas as gpd

# Load polygon of TABMON country boundaries (.shp)
tab = gpd.GeoDataFrame.from_file("SDM/tabmon_country_boundries/world-administrative-boundaries.shp")

# Calculate area and add column
tab["area"] = tab.area
tab = tab.drop([2]) # Removing Belgium row

# Generate and save html map
m = tab.explore()
output_file = "./polygon_map.html"
m.save(output_file)
