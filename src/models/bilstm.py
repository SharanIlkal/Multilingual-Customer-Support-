import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)
        scores = scores.squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        weights = weights.unsqueeze(-1)
        context = torch.sum(lstm_output * weights, dim=1)
        return context, weights


class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        padding_idx=0,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        context, attention_weights = self.attention(lstm_output)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits, attention_weights