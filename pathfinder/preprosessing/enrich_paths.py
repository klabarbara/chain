import os
import pandas as pd
import json

# File paths
JSONL_DIR = "../../data/processed_ndc_small"  # Directory containing multiple JSONL files
CSV_LOOKUP = "../../data/entities_with_coords.csv"
DRUG_LOOKUP = "../../data/products.csv"
OUTPUT_CSV = "../../data/path_with_identity.csv"

# Step 1: Load DEA Code Lookup Table (entities.csv) and Drug Lookup (products.csv)
lookup_df = pd.read_csv(CSV_LOOKUP)
lookup_dict = lookup_df.set_index("dea_no")[["lat", "lon"]].to_dict(orient="index")

drugs_df = pd.read_csv(DRUG_LOOKUP, dtype={"ndc": str})  # Ensure ndc is string
drugs_df["ndc"] = drugs_df["ndc"].str.strip()

# Step 2: Process All JSONL Files in the Directory
flattened_rows = []
path_counter = 1  # Counter to generate unique path IDs

for jsonl_file in os.listdir(JSONL_DIR):
    if jsonl_file.endswith(".jsonl"):  # Only process JSONL files
        jsonl_path = os.path.join(JSONL_DIR, jsonl_file)
        print(f"Processing {jsonl_path}...")

        with open(jsonl_path, "r") as file:
            for line in file:
                entry = json.loads(line.strip())
                path = entry["path"]
                quantity = entry["quantity"]
                ndc = str(entry["ndc"]).strip()  # Ensure ndc is string and clean

                # Generate a unique path ID
                path_id = f"path_{path_counter}_{ndc}"
                path_counter += 1

                # Generate consecutive from-to pairs and include step order
                for i in range(len(path) - 1):
                    from_dea = path[i]
                    to_dea = path[i + 1]
                    step = i + 1  # Step starts at 1

                    # Fetch lat/lon for from and to nodes
                    from_coords = lookup_dict.get(from_dea)
                    to_coords = lookup_dict.get(to_dea)

                    if from_coords and to_coords:
                        flattened_rows.append({
                            "path_id": path_id,
                            "ndc": ndc,
                            "step": step,
                            "from_dea_no": from_dea,
                            "to_dea_no": to_dea,
                            "from_lat": from_coords["lat"],
                            "from_lon": from_coords["lon"],
                            "to_lat": to_coords["lat"],
                            "to_lon": to_coords["lon"],
                            "quantity": quantity
                        })

# Step 3: Convert to DataFrame and Merge Drug Names
output_df = pd.DataFrame(flattened_rows)

# Merge drug names from the drugs_df using 'ndc' as the key
enriched_df = output_df.merge(drugs_df, on="ndc", how="left")

# Step 4: Save Enriched Data to CSV
enriched_df.to_csv(OUTPUT_CSV, index=False)

print(f"Enriched path data saved to '{OUTPUT_CSV}'.")
