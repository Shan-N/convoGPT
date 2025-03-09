config = {
    "MODEL_NAME": "gpt2",
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 5e-5,
    "EPOCHS": 3,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}
