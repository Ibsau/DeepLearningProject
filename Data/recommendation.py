import torch
import pandas as pd
from transformers import AutoTokenizer
from Classifier.Model import TransformerClassifier

def recommend_jobs(resume_text, jobs_dataframe, model, tokenizer, device, top_k=5, min_score=0.8):
    model.eval()
    recommendations = []

    for idx, row in jobs_dataframe.iterrows():
        job_desc = row['Job Description']
        macro_score = row['Macro_Score']

        # Tokenize
        encoding = tokenizer(
            job_desc,
            resume_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        macro_score_tensor = torch.tensor([macro_score], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, macro_score=macro_score_tensor)
            probs = torch.softmax(outputs, dim=1)
            match_prob = probs.squeeze()[1].item()

        if match_prob >= min_score:
            recommendations.append((job_desc, match_prob))

    # Sort recommendations
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_k]

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerClassifier(model_name="bert-base-uncased", num_classes=2)
model.load_state_dict(torch.load('/Users/shaiverma/Documents/CSE4095/DeepLearningProject/DeepLearningProject-2/DeepLearningProject/transformer_classifier_macro.pth', map_location=device))
model = model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load available job descriptions
jobs_df = pd.read_csv('/Users/shaiverma/Documents/CSE4095/DeepLearningProject/DeepLearningProject-2/DeepLearningProject/Data/test.jsonl')  # Or 'train_macro_2023.csv' depending

# Example resume
resume_text = "Experienced financial analyst with skills in investment banking, risk management, and corporate finance."

# Get recommendations
recommended_jobs = recommend_jobs(
    resume_text=resume_text,
    jobs_dataframe=jobs_df,
    model=model,
    tokenizer=tokenizer,
    device=device,
    top_k=5,
    min_score=0.95
)

# Show
print("\nTop Recommended Jobs:")
for idx, (desc, score) in enumerate(recommended_jobs):
    print(f"{idx+1}. Score: {score:.4f} | Job: {desc[:120]}...")