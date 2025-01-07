import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# File paths
INPUT_FILE = "data/entities_with_coords.jsonl"  # Input JSONL file
OUTPUT_FILE = "data/entities_with_coords.csv"   # Output CSV file
BATCH_SIZE = 5000              # Number of lines per batch
THREADS = 4                    # Number of threads to use

# Function to process a batch of JSON lines and convert to DataFrame
def process_batch(batch_lines):
    records = []
    for line in batch_lines:
        try:
            # Skip empty or whitespace-only lines
            if line.strip():
                records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping invalid line: {e}")
    return pd.DataFrame(records)

# Function to read JSONL file in chunks
def process_entities(input_file, output_file, batch_size, threads):
    # Open the input file and create the output CSV file with headers
    with open(input_file, 'r') as infile:
        # Read the first batch to initialize the CSV with headers
        batch = [next(infile) for _ in range(batch_size)]
        df = process_batch(batch)
        df.to_csv(output_file, index=False, mode='w', header=True)
        print(f"Initialized CSV with {len(df)} records.")

        # Process remaining batches with multithreading
        futures = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            while True:
                batch = list(infile.readline() for _ in range(batch_size))
                if not batch or not batch[0]:  # Stop if no more lines
                    break
                futures.append(executor.submit(process_batch, batch))

            # Write results to the CSV as batches complete
            for future in as_completed(futures):
                df_chunk = future.result()
                df_chunk.to_csv(output_file, index=False, mode='a', header=False)
                print(f"Appended {len(df_chunk)} records to CSV.")

    print(f"Conversion complete. Output saved to {output_file}")

# Run the conversion
process_entities(INPUT_FILE, OUTPUT_FILE, BATCH_SIZE, THREADS)
