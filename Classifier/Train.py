import torch
from transformers import AutoModel, AutoTokenizer
from Model import TransformerClassifier
import torch.nn as nn
import torch.optim as optim
from DatasetSampler import ResumeJobDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return tokens

model = TransformerClassifier(num_classes=2)  # Binary classification

# Sample input
text = "Experienced security and fire alarm installer with 5 years in the industry."
tokens = tokenize_text(text)

# Forward pass
logits = model(tokens["input_ids"], tokens["attention_mask"])
probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities

print(probabilities)  # Probability distribution over classes


# Loss function
criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)  # Transformer-friendly optimizer


# Test data
data = [
    {"Job-Description": "Security and Fire Alarm Installer...", "Resume": "Experienced installer...", "Label": 1},
    {"Job-Description": "Software Engineer...", "Resume": "Worked in sales...", "Label": 0},
]

# Create Dataset & DataLoader
train_dataset = ResumeJobDataset(data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


def train_model(model, train_loader, criterion, optimizer, device, epochs=3):
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()  # Reset gradients
            logits = model(input_ids, attention_mask)  # Forward pass

            loss = criterion(logits, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Track loss & accuracy
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)  # Convert logits to predicted class
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

# Train model
train_model(model, train_loader, criterion, optimizer, device, epochs=3)
