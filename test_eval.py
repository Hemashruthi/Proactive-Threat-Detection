import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from classifer import BERTClassifier
from dataset import TTPDataset
import json

# === Config ===
MODEL_PATH = "best_model.pt"  
TEST_DATA_PATH = "data/test_tokenized.json"
BATCH_SIZE = 24
MAX_LEN = 256
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Load label encoder ===
with open("data/label2id.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

# === Load test data ===
# with open(TEST_DATA_PATH, "r") as f:
#     test_data = json.load(f)

test_dataset = TTPDataset(TEST_DATA_PATH, label2id)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === Load model ===
model = BERTClassifier(num_labels=num_labels, dropout_prob=0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Evaluation ===
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
