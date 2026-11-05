import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # Encode all texts at once for speed
        encodings = tokenizer.encode_batch(texts)
        self.input_ids = [enc.ids for enc in encodings]
        
        # Store labels (assumed to be integers like 0, 1, 2)
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return a paired dictionary for this specific index
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class SmartDataCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __call__(self, batch):
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["label"] for item in batch]
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=self.pad_id 
        )
        
        stacked_labels = torch.stack(labels_list)
        return {
            "input_ids": padded_input_ids,
            "labels": stacked_labels
        }


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx = 0, function = nn.ReLU()):
        super(EncoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.ln = nn.LayerNorm(embedding_dim)
        self.hidden_ln = nn.LayerNorm([hidden_dim, hidden_dim])
        self.projection_0 = nn.Linear(embedding_dim, hidden_dim)
        self.projection_1 = nn.Linear(embedding_dim, hidden_dim)
        self.projection_final = nn.Linear(hidden_dim * hidden_dim, output_dim)

        self.function = function

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = self.ln(embedded)
        a = self.projection_0(embedded)
        b = self.projection_1(embedded)
        output = (a.transpose(-2, -1) @ b) / math.sqrt(input_ids.size(-1))
        output = self.function(output)
        output = self.hidden_ln(output)
        output = self.projection_final(output.view(output.size(0), -1))
        return output

class SkipConnection(nn.Module):
    def __init__(self, in_features, out_features, function=nn.ReLU()):
        super(SkipConnection, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_features, in_features),
            function,
            nn.Linear(in_features, out_features)
        )
        self.skip = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.skip(x) + self.module(x)

def evaluate_model(model, val_loader, criterion, device = "cpu"):
    model.eval()
    model.to(device)
    
    total_val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids) 
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels.float())
            total_val_loss += loss.item() * len(labels)
            
            predictions = (outputs > 0.5).long()
            
            correct_predictions += (predictions == labels.long()).sum().item()
            total_samples += len(labels)
            
    average_val_loss = total_val_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    return average_val_loss, accuracy

def train_model(model, optimizer, train_loader, num_epochs, criterion = nn.BCEWithLogitsLoss(), device = "cpu", val_loader = None):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
        
        avg_loss = total_loss / total_samples
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {avg_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {avg_loss:.4f}")

    return model

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Totale parameters: {total_params:,}")
    print(f"Trainbare parameters: {trainable_params:,}")