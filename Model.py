import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return tokens

class TransformerRegressor(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(TransformerRegressor, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.transformer.config.hidden_size, 1)  # Regression head

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = transformer_outputs.last_hidden_state[:, 0, :]  # Extract CLS token embedding
        # pooled_output = torch.mean(transformer_outputs.last_hidden_state, dim=1) # OPTIONAL Mean Pooling
        # pooled_output = torch.max(transformer_outputs.last_hidden_state, dim=1).values # OPtional Max Pooling

        score = self.regressor(pooled_output)  # Regression head
        return score.squeeze(-1)  # Remove extra dimensions

# Initialize the model
model = TransformerRegressor("bert-base-uncased")
