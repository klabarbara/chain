import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from keplergl import KeplerGl

df = pd.read_csv("../../test_data/path_with_identity_small.csv")

# Create a geometry column from lat/lon; shapely uses (lon, lat) ordering
df['geometry'] = df.apply(lambda row: Point(row['from_lon'], row['from_lat']), axis=1)

gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

geojson_str = gdf.to_json()

map_ = KeplerGl()
map_.add_data(data=geojson_str, name="Path Data")

map_.save_to_html(file_name="path_map_small.html")
print("Map saved as path_map_small.html")