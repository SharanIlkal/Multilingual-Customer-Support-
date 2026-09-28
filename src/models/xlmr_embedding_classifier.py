import torch
import torch.nn as nn


class XLMREmbeddingClassifier(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_classes=27,
        dropout=0.3,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim // 2,
                num_classes,
            ),
        )

    def forward(self, x):

        return self.network(x)