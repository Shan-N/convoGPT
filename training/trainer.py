import torch
from torch.optim import Adam
import sys
import os
import torch.nn as nn
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset_loader import get_dataloader
from models.transformer import TransformerChatbot

# Ensure script can import from parent directory

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix: Set padding token manually

# Load dataset
dataloader = get_dataloader("data/raw/dataset.jsonl", tokenizer, batch_size=4)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = TransformerChatbot().to(device)
optimizer = Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # ✅ Ignore padding in loss

# Training loop
epochs = 3
for epoch in range(epochs):
    total_loss = 0  # Track loss per epoch
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["input_ids"].to(device)  # GPT-2 uses input_ids as labels

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # ✅ Use model's built-in loss calculation

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "models/chatbot_model.pth")
print("Model saved successfully!")
