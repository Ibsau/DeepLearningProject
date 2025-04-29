import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Classifier (Matched or Unmatches)
class TransformerClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)  # Classification head

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask) # Transformer Encoder
        
        
        pooled_output = transformer_outputs.last_hidden_state[:, 0, :]  # Extract CLS token embedding
        pooled_output = torch.mean(transformer_outputs.last_hidden_state, dim=1) # OPTIONAL Mean Pooling
        # pooled_output = torch.max(transformer_outputs.last_hidden_state, dim=1).values # OPtional Max Pooling

        logits = self.classifier(pooled_output)  # Classification head
        
        return logits  # Raw logits (Softmax should be applied in training)
