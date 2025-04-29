import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from Model import TransformerClassifier
from DatasetSampler import ResumeJobDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_synthetic_data(path):
    """
    Load each record from a jsonl file and turn it into two examples:
      - one matched resume (label=1)
      - one unmatched resume (label=0)
    """
    examples = []
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            jd = rec["Job-Description"]
            examples.append({
                "Job-Description": jd,
                "Resume": rec["Resume-matched"],
                "Label": 1
            })
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
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total * 100

    # Compute additional metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    print(f"Eval Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

    return avg_loss, acc

if __name__ == "__main__":
    # 1. Load and prepare data
    train_examples = load_synthetic_data("dev.jsonl")
    val_examples   = load_synthetic_data("test.jsonl")

    train_dataset = ResumeJobDataset(train_examples, tokenizer)
    val_dataset   = ResumeJobDataset(val_examples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8)

    # 2. Build model, loss, optimizer
    model     = TransformerClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    # 3. Training loop
    epochs = 300
    best_val_loss = float('inf')  # initialize with infinity
    save_path = "best_model.pth"

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f},   Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"New best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
