import os
import json
from transformers import BertTokenizer
import pandas as pd
# Paths for input and output
DATA_FOLDER = "data"
FILE_MAP = {
    "train": "procedure_train.tsv",
    "dev": "procedure_dev.tsv",
    "test": "procedure_test.tsv"
}
MAX_LENGTH = 256
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to parse each TSV and tokenize
def process_and_tokenize(tsv_path):
    
    import ast

    df = pd.read_csv(tsv_path, sep="\t")
    tokenized_data = []

    for _, row in df.iterrows():
        sentence = str(row["text1"]).strip()
        try:
            labels = ast.literal_eval(row["labels"])
        except Exception:
            continue

        if sentence and isinstance(labels, list) and len(labels) > 0:
            first_label = labels[0].strip()
            if first_label.startswith("T"):
                encoding = tokenizer(
                    sentence,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors=None
                )
                tokenized_data.append({
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "label": first_label
                })

    return tokenized_data

# Process and save each file
for split, filename in FILE_MAP.items():
    file_path = os.path.join(DATA_FOLDER, filename)
    tokenized = process_and_tokenize(file_path)
    save_path = os.path.join(DATA_FOLDER, f"{split}_tokenized.json")
    with open(save_path, "w") as f:
        json.dump(tokenized, f, indent=2)

summary = {
    "Split": list(FILE_MAP.keys()),
    "Examples Tokenized": [len(process_and_tokenize(os.path.join(DATA_FOLDER, f))) for f in FILE_MAP.values()]
}

df = pd.DataFrame(summary)
print(df)
