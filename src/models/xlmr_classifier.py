import torch
import torch.nn as nn

from transformers import (
    AutoModel,
)


class XLMRClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_classes,
        dropout=0.3,
        freeze_encoder=True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        # Freeze the transformer initially
        if freeze_encoder:

            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # XLM-R last hidden state
        last_hidden_state = (
            outputs.last_hidden_state
        )

        # CLS / first token representation
        pooled_output = (
            last_hidden_state[:, 0, :]
        )

        pooled_output = self.dropout(
            pooled_output
        )

        logits = self.classifier(
            pooled_output
        )

        return logits