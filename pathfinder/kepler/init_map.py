import pandas as pd
from keplergl import KeplerGl

# Load the data
data = pd.read_csv("../../data/path_with_identity.csv")

# Ensure latitude and longitude columns are numeric
data['from_lat'] = pd.to_numeric(data['from_lat'], errors='coerce')
data['from_lon'] = pd.to_numeric(data['from_lon'], errors='coerce')
data['to_lat'] = pd.to_numeric(data['to_lat'], errors='coerce')
data['to_lon'] = pd.to_numeric(data['to_lon'], errors='coerce')

# Drop rows with missing or invalid lat/lon values
data = data.dropna(subset=['from_lat', 'from_lon', 'to_lat', 'to_lon'])

# Rename columns for Kepler
kepler_data = data.rename(columns={
    'from_lat': 'start_lat',
    'from_lon': 'start_lon',
    'to_lat': 'end_lat',
    'to_lon': 'end_lon'
})

# Initialize Kepler map
map_ = KeplerGl(height=800)

# Add data to Kepler
try:
    map_.add_data(data=kepler_data, name="Opioid Distribution Paths")
except Exception as e:
    print(f"Error adding data to Kepler: {e}")

# Save map to an HTML file
try:
    map_.save_to_html(file_name="opioid_distribution_paths.html")
    print("Map saved as opioid_distribution_paths.html")
except Exception as e:
    print(f"Error saving Kepler map: {e}")
