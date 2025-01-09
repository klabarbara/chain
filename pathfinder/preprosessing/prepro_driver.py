import enrich_paths, add_coords, add_transitions
# import add_ndc – Original ndc enrichment carried out on entire dataset in colab.
# Using small subset "processed_ndc_small" for development

import subprocess

def run_script(script_name, *args):
    """
    Runs a Python script as a subprocess with optional arguments.
    """
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(["python", script_name, *args], check=True, capture_output=True, text=True)
        print(f"{script_name} output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}:\n{e.stderr}")
        raise

def main():
    # Define the sequence of preprocessing steps
    preprocessing_steps = [
        ("add_coords.py", []),
        ("enrich_paths.py", []),
        ("add_transitions.py", [])
    ]

    # Run each step
    for script, args in preprocessing_steps:
        run_script(script, *args)

    print("Preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()