
import pandas as pd
import json
import time
from opencage.geocoder import OpenCageGeocode  

pd.set_option('display.max_columns', None)

ORIGINAL_ENTITIES_FILE = "../../data/entities.csv"  # original entities csv
ZIP_LL_FILE = "../../data/zip_ll.csv"              # zip-latlon dict downloaded from (source)
OUTPUT_JSONL_FILE = "../../data/entities_with_coords.jsonl"  

# Part 1: Mrging coords from zip_ll.csv into entities dataset

# Step 1: Load Original Entity Data
print("Loading original entity data...")
entities_df = pd.read_csv(ORIGINAL_ENTITIES_FILE, dtype={'zip': str})  # ensures 'zip' is read as string
print(f"Total entities loaded: {len(entities_df)}")

# Step 2: Load ZIP-to-Lat/Lon Data
print("Loading ZIP-to-lat/lon data...")
zip_ll_df = pd.read_csv(ZIP_LL_FILE, dtype={'ZIP': str})  
print(f"Total ZIP codes loaded: {len(zip_ll_df)}")

# Step 3: Clean and Prepare ZIP Codes
print("Cleaning ZIP codes in entities data...")
entities_df['zip'] = entities_df['zip'].str.strip().str.zfill(5)  # pad with leading zeros to ensure 5 digits

print("Cleaning ZIP codes in ZIP-to-lat/lon data...")
zip_ll_df['ZIP'] = zip_ll_df['ZIP'].str.strip().str.zfill(5)    

# Step 4: Merge Entity Data with ZIP Coordinates
print("Merging entity data with ZIP coordinates...")

zip_ll_df = zip_ll_df[['ZIP', 'LAT', 'LNG']]

# merge on ZIP code
merged_df = pd.merge(
    entities_df,
    zip_ll_df,
    left_on='zip',
    right_on='ZIP',
    how='left'
)

merged_df = merged_df.drop(columns=['ZIP'])

# Step 5: Handle Missing Coordinates
missing_coords = merged_df['LAT'].isnull().sum()
if missing_coords > 0:
    print(f"Warning: {missing_coords} entities have ZIP codes without corresponding coordinates.") # handled below
else:
    print("All entities have corresponding ZIP code coordinates.")

# Step 6: Export to JSON Lines
print(f"Exporting merged data to '{OUTPUT_JSONL_FILE}'...")

output_columns = ['dea_no', 'bus_act', 'city', 'state', 'zip', 'lat', 'lon']


merged_df = merged_df.rename(columns={'LAT': 'lat', 'LNG': 'lon'})


export_df = merged_df[output_columns]

# replace NaN with None for JSON compatibility
export_df = export_df.where(pd.notnull(export_df), None)

with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
    for _, row in export_df.iterrows():
        json_line = row.to_dict()
        f.write(json.dumps(json_line) + "\n")

print(f"Successfully saved '{OUTPUT_JSONL_FILE}'.")


# Part 2: Impute missing lat/lon values using geocoder. Not fetching all lat/lon values because of 
# free daily API limits. TODO: As such, will likely have to do over a couple days or cheese the last few manually.
# Any lat/lons not found will be input manually. Source to follow.
pd.set_option('display.max_columns', None)

ENTITIES_FILE = "../../data/entities_with_coords.jsonl"  
PROCESSED_DIR = "../../data/processed_ndc_small/"        # using subset of all paths for demo
OUTPUT_PATHS_FILE = "../../data/cleaned_distribution_paths.csv"
OUTPUT_ENTITIES_FILE = "../../data/cleaned_entities.csv"

GEOCODER_API_KEY = "c9340bed705d40bfa6a51720cff7146f"  
geocoder = OpenCageGeocode(GEOCODER_API_KEY)

def geocode_zip(zip_code):
    try:
        result = geocoder.geocode(f"{zip_code}, USA")
        if result:
            lat = result[0]['geometry']['lat']
            lon = result[0]['geometry']['lng']
            return lat, lon
    except Exception as e:
        print(f"Error geocoding {zip_code}: {e}")
    return None, None

# Step 1: Load entity data and filter for valid lat/lon
entities = pd.read_json(ENTITIES_FILE, lines=True)

# repad zips
entities['zip'] = entities['zip'].astype(str).str.zfill(5)

# Identify entities with missing lat/lon
missing_lat_lon = entities[entities[['lat', 'lon']].isna().any(axis=1)]

# geocode missing lat/lon for ZIP codes
for idx, row in missing_lat_lon.iterrows():
    zip_code = row['zip']
    lat, lon = geocode_zip(zip_code)
    if lat is not None and lon is not None:
        entities.at[idx, 'lat'] = lat
        entities.at[idx, 'lon'] = lon
    else:
        print(f"Unable to resolve ZIP code: {zip_code}")
    time.sleep(1)  # Prevent hitting rate limits for geocoding API

# Filter out entities that still have missing lat/lon
entities_valid = entities.dropna(subset=["lat", "lon"])

entities_valid = entities_valid.copy()

# Explicitly use .loc for column assignment
for col in ["dea_no", "bus_act", "city", "state", "zip"]:
    entities_valid.loc[:, col] = entities_valid[col].astype(str)

# Save the cleaned entities to a CSV file
entities_valid.to_csv(OUTPUT_ENTITIES_FILE, index=False)
print(f"Cleaned entities have been saved to '{OUTPUT_ENTITIES_FILE}'.")