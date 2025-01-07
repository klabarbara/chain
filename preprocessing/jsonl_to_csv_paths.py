import pandas as pd
import os
import glob

# File paths
ENTITIES_FILE = "/Users/karllabarbara/projects/chain/data/entities_with_coords.jsonl"  # Path to entity file
PROCESSED_DIR = "/Users/karllabarbara/projects/chain/data/processed_ndc_small/"      # Directory containing .jsonl files
OUTPUT_FILE = "/Users/karllabarbara/projects/chain/data/flattened_paths.csv"

# Step 1: Load entity data and create a DEA lookup dictionary
entities = pd.read_json(ENTITIES_FILE, lines=True)
dea_lookup = entities.set_index("dea_no")[["lat", "lon"]].to_dict(orient="index")

# Step 2: Process all .jsonl files in the 'processed/' directory
flattened_rows = []

# Iterate over all .jsonl files in the processed directory
for file in glob.glob(os.path.join(PROCESSED_DIR, "*.jsonl")):
    print(f"Processing file: {file}")
    paths = pd.read_json(file, lines=True)

    # Process each row in the .jsonl file
    for _, row in paths.iterrows():
        path = row['path']
        dates = row['dates']
        quantity = row['quantity']
        ndc = row['ndc']

        # Flatten consecutive path entries into from-to pairs
        for i in range(len(path) - 1):
            from_dea = path[i]
            to_dea = path[i + 1]
            date = dates[i] if i < len(dates) else None  # Handle dates safely

            # Lookup coordinates for 'from' and 'to' DEA codes
            from_coords = dea_lookup.get(from_dea, {"lat": None, "lon": None})
            to_coords = dea_lookup.get(to_dea, {"lat": None, "lon": None})

            # Append the flattened row
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

# Step 3: Save flattened data to a CSV file
flattened_df = pd.DataFrame(flattened_rows)
flattened_df.to_csv(OUTPUT_FILE, index=False)

print(f"Flattened data has been saved to '{OUTPUT_FILE}'")
