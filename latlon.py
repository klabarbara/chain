import csv
import json
import requests
import time

def get_lat_lon_nominatim(zipcode, country="US"):
    # Nominatim endpoint
    url = "https://nominatim.openstreetmap.org/search"
    # Parameters for the search
    params = {
        'postalcode': zipcode,
        'country': country,
        'format': 'json',
        'addressdetails': 0,
        'limit': 1
    }
    # Include a custom User-Agent to comply with usage policy
    headers = {
        'User-Agent': 'MyApp/1.0 (myemail@example.com)' 
    }

    response = requests.get(url, params=params, headers=headers)
    # Handle response errors gracefully
    if response.status_code != 200:
        return None, None
    
    data = response.json()
    if data:
        lat = data[0].get('lat', None)
        lon = data[0].get('lon', None)
        if lat and lon:
            return float(lat), float(lon)
    return None, None

input_csv = 'data/entities_updated.csv'  # Replace with your actual CSV file path
output_jsonl = 'data/entities_with_coords.jsonl'

# A cache to avoid multiple lookups for the same ZIP code
zip_cache = {}

with open(input_csv, 'r', newline='', encoding='utf-8') as fin, \
     open(output_jsonl, 'w', encoding='utf-8') as fout:

    reader = csv.DictReader(fin)
    for row in reader:
        zipcode = row['zip']
        if zipcode not in zip_cache:
            lat, lon = get_lat_lon_nominatim(zipcode)
            # Cache the result (even if None, None, to avoid retrying too often)
            zip_cache[zipcode] = (lat, lon)
            # Optional: respect Nominatim rate limits by sleeping briefly
            # time.sleep(1)  # Consider adding a delay if needed
        else:
            lat, lon = zip_cache[zipcode]

        # Include lat/lon in the output entry
        entry = {
            "dea_no": row["dea_no"],
            "bus_act": row["bus_act"],
            "city": row["city"],
            "state": row["state"],
            "zip": row["zip"],
            "lat": lat,
            "lon": lon
        }

        fout.write(json.dumps(entry) + "\n")
        # Print for progress
        print("Processed:", entry)
