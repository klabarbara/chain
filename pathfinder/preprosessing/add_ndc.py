import os
import json
from tqdm.auto import tqdm


def add_ndc_to_records(input_dir, output_dir):
    """
    Reads JSONL files, adds the NDC from the filename to each record,
    and saves the updated records to a new JSONL file.

    Parameters:
    - input_dir: Directory containing the processed JSONL files.
    - output_dir: Directory to save the updated JSONL files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(os.listdir(input_dir), desc="Processing files"):
        if file.startswith("paths_") and file.endswith(".jsonl"):
            # Extract NDC from the filename
            ndc = file.replace("paths_", "").replace(".jsonl", "")
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file)

            with open(input_file, "r") as infile, open(output_file, "w") as outfile:
                for line in infile:
                    record = json.loads(line)
                    record["ndc"] = ndc  # Add the NDC field
                    outfile.write(json.dumps(record) + "\n")

            print(f"Updated file saved to: {output_file}")









input_directory = "../../data/processed" 
output_directory = "../../data/processed_with_ndc"
add_ndc_to_records(input_directory, output_directory)