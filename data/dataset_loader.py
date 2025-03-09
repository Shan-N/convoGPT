import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
        
        self.df = pd.DataFrame(data)

        if "input persona" not in self.df.columns or "synthesized text" not in self.df.columns:
            raise ValueError("Dataset must have 'input persona' and 'synthesized text' columns.")

        self.tokenizer = tokenizer  # ✅ Accept tokenizer object instead of reloading it
        self.tokenizer.pad_token = self.tokenizer.eos_token  # ✅ Ensure padding token is set

        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        persona = self.df.iloc[idx]["input persona"]
        synthesized_text = self.df.iloc[idx]["synthesized text"]

        text = f"Persona: {persona} | Generated Tool: {synthesized_text}"

        encoding = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        return {key: val.squeeze(0) for key, val in encoding.items()}

def get_dataloader(file_path, tokenizer, batch_size=8):
    dataset = ChatDataset(file_path, tokenizer)  # ✅ Pass tokenizer object
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # ✅ Set padding token

    dataset = ChatDataset("data/raw/dataset.jsonl", tokenizer)
    print(dataset.df.head())
