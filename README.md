# Transformer Chatbot Training  

This repository contains the training pipeline for a **Transformer-based Chatbot** using PyTorch and the Hugging Face `transformers` library. The chatbot is based on **GPT-2** and trained on a dataset of synthetic conversations.  

---

## 🚀 **Project Structure**  

```
oCray/
│── data/  
│   ├── dataset_loader.py  # Loads and preprocesses the dataset  
│   ├── raw/dataset.jsonl  # The dataset (JSONL format)  
│  
│── models/  
│   ├── transformer.py  # TransformerChatbot model  
│   ├── chatbot_model.pth  # Trained model weights (after training)  
│  
│── training/  
│   ├── trainer.py  # Training script  
│  
│── README.md  # This file  
```

---

## 📂 **Dataset Format**  
The dataset is stored in **JSONL (JSON Lines) format** at `data/raw/dataset.jsonl`.  
Each line is a separate JSON object with the following structure:  

```json
{"input persona": "Friendly AI", "synthesized text": "Hello! How can I assist you today?"}
```

- **`input persona`**: The persona or character of the chatbot.  
- **`synthesized text`**: The chatbot's response based on that persona.  

---

## 🛠 **How It Works**  

### 1️⃣ **Dataset Loader (`data/dataset_loader.py`)**  
- Loads the dataset from `dataset.jsonl`  
- Uses a tokenizer to convert text into tokenized tensors  
- Returns a PyTorch `DataLoader` for batching  

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token for GPT-2

dataloader = get_dataloader("data/raw/dataset.jsonl", tokenizer, batch_size=4)
```

---

### 2️⃣ **Model Definition (`models/transformer.py`)**  
- The chatbot is implemented in `TransformerChatbot` (based on GPT-2).  
- If you're using a custom model, ensure it returns logits of shape `(batch_size, seq_len, vocab_size)`.  

---

### 3️⃣ **Training Script (`training/trainer.py`)**  
The script:  
✅ Loads the dataset  
✅ Initializes the GPT-2 model  
✅ Trains for **3 epochs** using `Adam` optimizer  
✅ Saves the trained model at `models/chatbot_model.pth`  

#### 🔹 **Training Loop**  
```python
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["input_ids"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Model's built-in loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")
```

---

## 🏃‍♂️ **How to Train the Model**  

### 1️⃣ **Install Dependencies**  
Make sure you have **Python 3.8+** and install required libraries:  
```bash
pip install torch transformers pandas
```

### 2️⃣ **Run the Training Script**  
```bash
python training/trainer.py
```

### 3️⃣ **Save and Load Model**  
After training, the model is saved at:  
```
models/chatbot_model.pth
```
To load the model for inference:  
```python
model.load_state_dict(torch.load("models/chatbot_model.pth"))
model.eval()  # Set to evaluation mode
```

---

## 🤖 **Next Steps**  
- Improve dataset quality for better chatbot responses  
- Fine-tune on larger conversation datasets  
- Deploy the chatbot using FastAPI or Flask  

---

## 🐝 **License**  
This project is open-source under the **MIT License**.  
Feel free to modify and improve! 🚀

