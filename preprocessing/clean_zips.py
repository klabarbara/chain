import pandas as pd
import os
import glob

# File paths
ENTITIES_FILE = "data/entities_with_coords.jsonl"  # Path to entity file
PROCESSED_DIR = "data/processed_ndc_small/"      # Directory containing .jsonl files
OUTPUT_PATHS_FILE = "cleaned_distribution_paths.csv"
OUTPUT_ENTITIES_FILE = "cleaned_entities.csv"

# Step 1: Load entity data and filter for valid lat/lon
entities = pd.read_json(ENTITIES_FILE, lines=True)
# re-padding zip codes with missing leading zeros
entities['zip'] = entities['zip'].astype(str)
entities['zip'] = entities['zip'].str.zfill(5)
print(entities['zip'].head())
print(f'{len(entities)}\n')
entities_valid = entities.dropna(subset=["lat", "lon"])  # Remove entities with missing lat/lon
entities_valid.loc[:,"dea_no"] = entities_valid["dea_no"].astype(str)
entities_valid.loc[:,"bus_act"] = entities_valid["bus_act"].astype(str)
entities_valid.loc[:,"city"] = entities_valid["city"].astype(str)
entities_valid.loc[:,"state"] = entities_valid["state"].astype(str)
entities_valid.loc[:,"zip"] = entities_valid["zip"].astype(str)

# Save the cleaned entities to a CSV file
entities_valid.to_csv(OUTPUT_ENTITIES_FILE, index=False)
print(f"Cleaned entities have been saved to '{OUTPUT_ENTITIES_FILE}'.")

# Create a lookup dictionary for valid DEA locations
dea_lookup = entities_valid.set_index("dea_no")[["lat", "lon"]].to_dict(orient="index")

# Step 2: Process all .jsonl files in the 'processed/' directory
cleaned_rows = []

# Iterate over all .jsonl files in the processed directory
for file in glob.glob(os.path.join(PROCESSED_DIR, "*.jsonl")):
    # print(f"Processing file: {file}")
    paths = pd.read_json(file, lines=True)

    # Process each row in the .jsonl file
    for _, row in paths.iterrows():
        path = row['path']
        dates = row['dates']
        quantity = row['quantity']
        ndc = row['ndc']

        # Flatten consecutive path entries into from-to pairs
        for i in range(len(path) - 1):
            print(path)
            from_dea = path[i]
            to_dea = path[i + 1]
            date = dates[i] if i < len(dates) else None  # Handle dates safely

            # Skip paths where 'from' or 'to' DEA codes are not in the lookup
            if from_dea not in dea_lookup or to_dea not in dea_lookup:
                # print(entities_valid.dtypes)
                # print(type(from_dea), type(to_dea))
                print(f'{from_dea=}   from zip: {entities_valid[entities_valid["dea_no"] == from_dea]["zip"]}\n{to_dea=}    to zip: {entities_valid[entities_valid["dea_no"] == to_dea]["zip"]}\n\n')
                continue

            # Lookup coordinates for 'from' and 'to' DEA codes
            from_coords = dea_lookup[from_dea]
            to_coords = dea_lookup[to_dea]

            # Append the cleaned row
            cleaned_rows.append({
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

# Step 3: Save cleaned paths to a CSV file
cleaned_df = pd.DataFrame(cleaned_rows)
cleaned_df.to_csv(OUTPUT_PATHS_FILE, index=False)
print(f'{len(cleaned_df)=}')
print(f"Cleaned distribution paths have been saved to '{OUTPUT_PATHS_FILE}'.")
