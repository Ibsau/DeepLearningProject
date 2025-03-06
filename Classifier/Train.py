import torch
from transformers import AutoModel, AutoTokenizer
from Model import TransformerClassifier

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
