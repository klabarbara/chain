import pandas as pd

# Load JSONL files
entities = pd.read_json("entities.jsonl", lines=True)
paths = pd.read_json("paths.jsonl", lines=True)

# Create a lookup dictionary for DEA locations (lat/lon)
dea_lookup = entities.set_index("dea_no")[["lat", "lon"]].to_dict(orient="index")

# Initialize a list to store the flattened rows
flattened_rows = []

# Process each path entry
for _, row in paths.iterrows():
    path = row['path']
    dates = row['dates']
    quantity = row['quantity']
    ndc = row['ndc']
    
    # Iterate over consecutive DEA code pairs in the path
    for i in range(len(path) - 1):
        from_dea = path[i]
        to_dea = path[i + 1]
        date = dates[i] if i < len(dates) else None  # Handle dates
        
        # Lookup lat/lon for both "from" and "to" DEA codes
        from_coords = dea_lookup.get(from_dea, {"lat": None, "lon": None})
        to_coords = dea_lookup.get(to_dea, {"lat": None, "lon": None})
        
        # Append flattened row to list
        flattened_rows.append({
            "ndc": ndc,
            "from_dea_no": from_dea,
            "to_dea_no": to_dea,
            "date": date,
            "quantity": quantity,
            "from_lat": from_coords["lat"],
            "from_lon": from_coords["lon"],
            "to_lat": to_coords["lat"],
            "to_lon": to_coords["lon"]
        })

# Convert to DataFrame
flattened_df = pd.DataFrame(flattened_rows)

# Save to CSV
flattened_df.to_csv("flattened_distribution_paths.csv", index=False)

# Print confirmation
print("Flattened data has been saved to 'flattened_distribution_paths.csv'.")
