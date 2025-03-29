import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import json

def jsonl_to_dict(file_path):
    """Converts a JSONL file to a dictionary.

    Args:
        file_path: The path to the JSONL file.

    Returns:
        A dictionary containing the data from the JSONL file.
    """
    data_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_object = json.loads(line)
                # Assuming each JSON object has a unique key
                # Modify this part based on your JSON structure
                if isinstance(json_object, dict):
                  key = list(json_object.keys())[0]
                  data_dict[key] = json_object[key]
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data_dict

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class ResumeJobDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data  # List of dictionaries with job description, resume, and label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        job_desc = self.data[idx]["Job-Description"]
        resume = self.data[idx]["Resume"]
        label = self.data[idx]["Label"]  # 0 = Unmatched, 1 = Matched (Binary)

        # Combine job description and resume for similarity context
        combined_text = f"Job: {job_desc} [SEP] Resume: {resume}"

        # Tokenize input text
        tokens = self.tokenizer(
            combined_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)  # Remove batch dim
        attention_mask = tokens["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)  # LongTensor for classification
        }


if __name__ == '__main__':
    dataset_path = "Datasets/dev.jsonl"
    dataset = jsonl_to_dict(dataset_path)