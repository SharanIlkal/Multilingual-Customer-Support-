import os
import re
import torch
import torch.nn as nn

MODEL_PATH = "models/bilstm_attention.pt"
MAX_LENGTH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTENT_TO_CATEGORY = {
    "cancel_order": "ORDER",
    "change_order": "ORDER",
    "change_shipping_address": "SHIPPING",
    "check_cancellation_fee": "CANCEL",
    "check_invoice": "INVOICE",
    "check_payment_methods": "PAYMENT",
    "check_refund_policy": "REFUND",
    "complaint": "FEEDBACK",
    "contact_customer_service": "CONTACT",
    "contact_human_agent": "CONTACT",
    "create_account": "ACCOUNT",
    "delete_account": "ACCOUNT",
    "delivery_options": "DELIVERY",
    "delivery_period": "DELIVERY",
    "edit_account": "ACCOUNT",
    "get_invoice": "INVOICE",
    "get_refund": "REFUND",
    "newsletter_subscription": "SUBSCRIPTION",
    "payment_issue": "PAYMENT",
    "place_order": "ORDER",
    "recover_password": "ACCOUNT",
    "registration_problems": "ACCOUNT",
    "review": "FEEDBACK",
    "set_up_shipping_address": "SHIPPING",
    "switch_account": "ACCOUNT",
    "track_order": "ORDER",
    "track_refund": "REFUND",
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def tokenize(text):
    text = str(text).lower().strip()
    return re.findall(r"\b\w+\b",text)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output, mask,):
        scores = self.attention(lstm_output).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights

class BiLSTMAttentionClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        num_layers=2,
        dropout=0.3,
        pad_idx=0,
    ):

        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        context, attention_weights = (self.attention(lstm_output, mask))
        context = self.dropout(context)
        logits = self.classifier(context)
        return (logits, attention_weights)

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    vocab = checkpoint["vocab"]
    label_to_id = checkpoint["label_to_id"]
    id_to_label = checkpoint["id_to_label"]
    config = checkpoint["config"]

    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=len(label_to_id),
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_idx=vocab[PAD_TOKEN],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return (
        model,
        vocab,
        label_to_id,
        id_to_label,
    )

def encode_text(
    text,
    vocab,
    max_length=MAX_LENGTH,
):

    tokens = tokenize(text)
    token_ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    token_ids = token_ids[:max_length]

    if len(token_ids) < max_length:
        token_ids += [vocab[PAD_TOKEN]] * (max_length - len(token_ids))
    return token_ids

@torch.no_grad()
def predict(
    text,
    model,
    vocab,
    id_to_label,
    top_k=3,
):

    input_ids = encode_text(text,vocab,)
    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    logits, attention_weights = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1,)
    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            k=min(top_k, probabilities.size(1)),
            dim=1,
        )
    )

    predictions = []
    for probability, index in zip(
        top_probabilities[0],
        top_indices[0],
    ):

        intent = id_to_label[index.item()]
        category = INTENT_TO_CATEGORY.get(intent, "UNKNOWN")
        predictions.append(
            {
                "intent": intent,
                "category": category,
                "confidence": probability.item(),
            }
        )

    return (
        predictions,
        attention_weights,
    )

def display_predictions(predictions,):
    print("\nPredictions")
    print("-" * 60)
    for rank, prediction in enumerate(predictions, start=1,):
        print(f"{rank}. " f"{prediction['intent']}")
        print(f"   Category   : " f"{prediction['category']}")
        print(f"   Confidence : " f"{prediction['confidence'] * 100:.2f}%")

def main():
    print("=" * 70)
    print("MULTILINGUAL CUSTOMER SUPPORT")
    print("BiLSTM + ATTENTION INFERENCE")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print("\nLoading trained model...")
    (
        model,
        vocab,
        label_to_id,
        id_to_label,
    ) = load_model()

    print(f"Vocabulary size: " f"{len(vocab):,}")
    print(f"Number of intents: " f"{len(label_to_id)}")
    print("\nModel loaded successfully.")
    print("\nType 'exit' to stop.")
    while True:
        print()
        text = input("Customer message: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            print("Please enter a message.")
            continue
        (predictions, _, ) = predict(
            text,
            model,
            vocab,
            id_to_label,
            top_k=3,
        )
        display_predictions(predictions)

if __name__ == "__main__":
    main()