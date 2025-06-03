import torch
from torch.utils.data import Dataset
import json

class TTPDataset(Dataset):
    def __init__(self, json_path, label2id):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        label = torch.tensor(self.label2id[item['label']], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }