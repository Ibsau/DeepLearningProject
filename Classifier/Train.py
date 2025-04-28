import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from Model import TransformerClassifier
from DatasetSampler import ResumeJobDataset
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_synthetic_data(path):
    """
    Load each record from dev.jsonl and turn it into two examples:
      - one matched resume (label=1)
      - one unmatched resume (label=0)
    """
    examples = []
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            jd = rec["Job-Description"]
            # matched example
            examples.append({
                "Job-Description": jd,
                "Resume": rec["Resume-matched"],
                "Label": 1
            })
            # unmatched example
            examples.append({
                "Job-Description": jd,
                "Resume": rec["Resume-unmatched"],
                "Label": 0
            })
    return examples

def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total * 100
    return avg_loss, acc

@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total * 100
    return avg_loss, acc

if __name__ == "__main__":
    # 1. Load and prepare data
    raw_examples = load_synthetic_data(r"C:\Users\Benja\Documents\Projects\DeepLearningProj\DeepLearningProject\dev.jsonl")
    dataset = ResumeJobDataset(raw_examples, tokenizer)

    # 2. Split into train / val
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8)

    # 3. Build model, loss, optimizer
    model     = TransformerClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    # 4. Training loop
    epochs = 3
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate_model(model, val_loader,   criterion, device)
        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f},   Val Acc: {val_acc:.2f}%")
