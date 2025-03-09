import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class TransformerChatbot(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(TransformerChatbot, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))

        return type('TransformerOutput', (), {
            'loss': loss,
            'logits': logits
        })()

