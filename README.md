Model Architecture:
1. Convert text into tokens using tokenizer
2. Transformer Encoder to convert embeddings for each token
3. Pooling layer (Mean or Max pooling or [CLS] token embedding)
4. Regression Head to convert pooling representation into final score 
