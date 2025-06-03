import json
from collections import defaultdict

def build_label_encoder(json_paths):
    all_labels = set()

    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
            for example in data:
                all_labels.add(example["label"])

    label_list = sorted(all_labels)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    print(f"🔢 Found {len(label2id)} unique labels.")
    return label2id, id2label

if __name__ == "__main__":
    paths = [
        "data/train.json",
        "data/dev.json",
        "data/test.json"
    ]
    label2id, id2label = build_label_encoder(paths)

    # Save to file (optional)
    with open("data/label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open("data/id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print("label2id example:", list(label2id.items())[:5])
