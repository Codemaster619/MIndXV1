# train.py
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import Trainer, TrainingArguments
import numpy as np
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler

# Load data from JSON file
with open('primate_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Tokenizer and model initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=len(data[0]["annotations"]))

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Custom Datasets and DataLoaders
from dataset_Class import CustomDataset

train_dataset = CustomDataset(train_data, tokenizer, max_length=512)
val_dataset = CustomDataset(val_data, tokenizer, max_length=512)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Custom Data Collator
def custom_data_collator(batch):
    inputs = {key: torch.stack([sample[key] for sample in batch]) for key in batch[0].keys() if key != 'labels'}
    inputs['labels'] = torch.stack([sample['labels'] for sample in batch])
    return inputs

# Training loop using Trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * 3,
)
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    pred_probs = 1 / (1 + np.exp(-predictions))  # Sigmoid to get probabilities
    pred_labels = (pred_probs >= 0.5).astype(int)

    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, average='weighted')
    recall = recall_score(labels, pred_labels, average='weighted')
    f1 = f1_score(labels, pred_labels, average='weighted')
    roc_auc = roc_auc_score(labels, pred_probs, average='weighted', multi_class='ovr')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'binary_cross_entropy': p.loss,
    }

training_args.compute_metrics = compute_metrics

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('./bert_base_cased_model')
