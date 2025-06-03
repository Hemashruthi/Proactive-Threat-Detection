import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_scheduler
from torch.optim import AdamW
from classifer import BERTClassifier
from dataset import TTPDataset
import json
import os
from sklearn.metrics import accuracy_score

# Hyperparameters
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 30
MAX_LENGTH = 256

# File paths
TRAIN_FILE = "data/train_tokenized.json"
VAL_FILE = "data/dev_tokenized.json"
LABEL_ENCODER_PATH = "data/label2id.json"

# Load label encoder
with open(LABEL_ENCODER_PATH, 'r') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# Load datasets
dev_dataset = TTPDataset(VAL_FILE, label2id)
train_dataset = TTPDataset(TRAIN_FILE, label2id)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BERTClassifier(num_labels=len(label2id)).to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Loss function
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_loader:
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels = torch.tensor(batch["label"]).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {train_acc:.4f}")

    # Evaluation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"\tValidation Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print("\t🎯 Best model saved.")

# Final message
print("\nTraining complete. Best model saved as 'best_model.pt'")
