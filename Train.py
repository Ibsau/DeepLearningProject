from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from Model import tokenizer, tokenize_text, TransformerRegressor

class TextDataset(Dataset):
    def __init__(self, texts, scores):
        self.texts = texts
        self.scores = scores

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize_text(self.texts[idx])
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0), score


# Initialize the model
model = TransformerRegressor("bert-base-uncased")

# Example dataset
texts = ["This is a great movie!", "I don't like this book.", "The food was amazing."]
scores = [85.0, 20.0, 95.0]  # Scores between 0-100

dataset = TextDataset(texts, scores)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # Train for 3 epochs
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
