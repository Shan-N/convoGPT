import torch
from transformers import AutoTokenizer
from models.transformer import TransformerChatbot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TransformerChatbot().to(device)
model.load_state_dict(torch.load("models/chatbot_model.pth"))
model.eval()

def generate_response(persona):
    inputs = tokenizer(f"Persona: {persona}", return_tensors="pt").to(device)
    output = model.model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        persona_input = input("Enter Persona: ")
        if persona_input.lower() in ["exit", "quit"]:
            break
        print("AI Generated Tool:", generate_response(persona_input))
