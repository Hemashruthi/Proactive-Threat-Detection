import os
import pandas as pd
import json
import ast

# Input and output paths
DATASET_PATH = "data"  # Directory where TSV files are stored
SAVE_PATHS = {
    "train": "data/train.json",
    "dev": "data/dev.json",
    "test": "data/test.json"
}

# Function to process a single TSV file and extract examples
def process_file(file_path):
    df = pd.read_csv(file_path, sep="\t")
    processed = []

    for _, row in df.iterrows():
        sentence = str(row["text1"]).strip()
        try:
            labels = ast.literal_eval(row["labels"])  # Convert stringified list to actual list
        except Exception:
            continue  # Skip malformed entries

        if sentence and isinstance(labels, list) and len(labels) > 0:
            first_label = labels[0].strip()
            if first_label.startswith("T"):
                processed.append({
                    "sentence": sentence,
                    "label": first_label
                })
    return processed

# Process each split and save to separate JSON files
result_counts = {}
for split, output_path in SAVE_PATHS.items():
    file_path = os.path.join(DATASET_PATH, f"procedure_{split}.tsv")
    examples = process_file(file_path)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)
    result_counts[split] = len(examples)

result_counts
