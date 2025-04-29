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

if __name__ == "__main__":
    # Paths
    checkpoint_path = "best_model.pth"
    data_path = "dev.jsonl"  # Assumes dev.jsonl is in the same folder

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

    # Predict
    pred_label, confidence = predict_match(model, job_desc, resume)

    # Results
    print("\n===== Evaluation Results =====")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {pred_label} (Confidence: {confidence:.2f})")
