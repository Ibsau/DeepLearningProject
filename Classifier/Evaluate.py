import torch
import json
import random
from transformers import AutoTokenizer
from Model import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_synthetic_data(path):
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

def prepare_input(job_desc, resume, tokenizer, max_length=512):
    combined_text = f"Job: {job_desc} [SEP] Resume: {resume}"

    tokens = tokenizer(
        combined_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    return input_ids, attention_mask

def predict_match(model, job_desc, resume):
    model.eval()
    input_ids, attention_mask = prepare_input(job_desc, resume, tokenizer)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()

    label_str = "Matched" if pred_label == 1 else "Unmatched"
    return label_str, confidence

def recommend_top_jobs(model, resume_text, job_desc_list, top_k=5):
    model.eval()
    scores = []

    for jd in job_desc_list:
        input_ids, attention_mask = prepare_input(jd, resume_text, tokenizer)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            match_prob = probs[0, 1].item()  # Probability of being "Matched" (label 1)

        scores.append((jd, match_prob))

    # Sort job descriptions by match probability descending
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    return top_matches

if __name__ == "__main__":
    # Paths
    checkpoint_path = "/Users/shaiverma/Documents/CSE4095/DeepLearningProject/DeepLearningProject-2/DeepLearningProject/best_model-2.pth"
    data_path = "/Users/shaiverma/Documents/CSE4095/DeepLearningProject/DeepLearningProject-2/DeepLearningProject/Data/dev.jsonl"  # Assumes dev.jsonl is in the same folder

    # Load model
    model = TransformerClassifier(num_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load dataset
    raw_examples = load_synthetic_data(data_path)

    # Sample one random example
    sample = random.choice(raw_examples)
    job_desc = sample["Job-Description"]
    resume = sample["Resume"]
    true_label = "Matched" if sample["Label"] == 1 else "Unmatched"

    print("\n===== Sampled Job Description =====")
    print(job_desc)
    print("\n===== Sampled Resume =====")
    print(resume)

    # Predict match on sampled pair
    pred_label, confidence = predict_match(model, job_desc, resume)

    print("\n===== Match Evaluation =====")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {pred_label} (Confidence: {confidence:.2f})")

    # Generate Top 5 Recommendations
    job_descs = list({ex["Job-Description"] for ex in raw_examples})  # deduplicated JDs
    top_jobs = recommend_top_jobs(model, resume, job_descs, top_k=5)

    print("\n===== Top 5 Job Recommendations for This Resume =====")
    for rank, (jd, score) in enumerate(top_jobs, start=1):
        print(f"\n--- Rank #{rank} (Match Score: {score:.2f}) ---")
        print(jd)
