import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from keplergl import KeplerGl

# 1) Load CSV into a regular pandas DataFrame
df = pd.read_csv("../../data/path_with_identity_small.csv")

# 2) Create a geometry column from lat/lon
#    Note: shapely uses (lon, lat) ordering
df['geometry'] = df.apply(lambda row: Point(row['from_lon'], row['from_lat']), axis=1)

# 3) Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# 4) Convert GeoDataFrame to a GeoJSON string
geojson_str = gdf.to_json()

# 5) Pass this GeoJSON string to Kepler
map_ = KeplerGl()
map_.add_data(data=geojson_str, name="Path Data")

map_.save_to_html(file_name="path_map.html")
print("Map saved as path_map.html")