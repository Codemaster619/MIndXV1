# custom_dataset.py
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        post_text = self.data[index]["post_text"]
        labels = [int(label[1] == "yes") for label in self.data[index]["annotations"]]

        encoding = self.tokenizer(
            post_text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length'
        )
        inputs = {key: torch.squeeze(value) for key, value in encoding.items()}
        inputs['labels'] = torch.tensor(labels, dtype=torch.float32)

        return inputs
