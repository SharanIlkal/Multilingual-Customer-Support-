import os
import random
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,f1_score,)
from torch.utils.data import Dataset, DataLoader

SEED = 42
TRAIN_PATH = "data/processed/final/train.csv"
VAL_PATH = "data/processed/final/validation.csv"
TEST_PATH = "data/processed/final/test.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_attention.pt")
MAX_VOCAB_SIZE = 30000
MIN_FREQ = 2
MAX_LENGTH = 32
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.30
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize(text):
    text = str(text).lower().strip()
    return re.findall(r"\b\w+\b", text)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab(texts):
    word_counts = {}
    for text in texts:
        for token in tokenize(text):
            word_counts[token] = (word_counts.get(token, 0) + 1)
    sorted_words = sorted(
        word_counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, count in sorted_words:
        if count < MIN_FREQ:
            continue
        if len(vocab) >= MAX_VOCAB_SIZE:
            break
        vocab[word] = len(vocab)
    return vocab

def build_label_mapping(train_labels):
    labels = sorted(train_labels.unique())
    label_to_id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id_to_label = {
        idx: label
        for label, idx in label_to_id.items()
    }
    return label_to_id, id_to_label

class TextDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocab,
        label_to_id,
        max_length,
    ):

        self.texts = (
            dataframe["text"]
            .astype(str)
            .tolist()
        )

        self.labels = [
            label_to_id[label]
            for label in dataframe["intent"]
        ]

        self.vocab = vocab
        self.max_length = max_length

    def encode_text(self, text):
        tokens = tokenize(text)
        token_ids = [
            self.vocab.get(token, self.vocab[UNK_TOKEN])
            for token in tokens
        ]

        token_ids = token_ids[:self.max_length]
        if len(token_ids) < self.max_length:
            token_ids += [self.vocab[PAD_TOKEN]] * (self.max_length - len(token_ids))
        return token_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.encode_text(self.texts[idx])
        return (
            torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            torch.tensor(
                self.labels[idx],
                dtype=torch.long
            ),
        )

class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output,mask,):
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
        return (logits,attention_weights)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
        )
        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
        )
    avg_loss = (total_loss / len(dataloader))
    accuracy = accuracy_score(all_labels, all_predictions)
    return (avg_loss, accuracy)

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        logits, _ = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels
            .cpu()
            .numpy()
        )

    avg_loss = (total_loss/ len(dataloader))
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        all_labels,
        all_predictions,
    )

def main():
    set_seed()
    print("=" * 70)
    print("BiLSTM + ATTENTION CLASSIFIER")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print("\nLoading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(val_df):,}")
    print(f"Test       : {len(test_df):,}")
    label_to_id, id_to_label = (build_label_mapping(train_df["intent"]))
    num_classes = len(label_to_id)
    print(
        f"\nNumber of classes: "
        f"{num_classes}"
    )

    print("\nBuilding vocabulary...")
    vocab = build_vocab(train_df["text"])
    print(
        f"Vocabulary size: "
        f"{len(vocab):,}"
    )

    train_dataset = TextDataset(
        train_df,
        vocab,
        label_to_id,
        MAX_LENGTH,
    )
    val_dataset = TextDataset(
        val_df,
        vocab,
        label_to_id,
        MAX_LENGTH,
    )
    test_dataset = TextDataset(
        test_df,
        vocab,
        label_to_id,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab[PAD_TOKEN],
    )

    model = model.to(DEVICE)
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("\nModel configuration:")
    print(f"Embedding dimension : " f"{EMBED_DIM}")
    print(f"Hidden dimension    : " f"{HIDDEN_DIM}")
    print(f"LSTM layers         : " f"{NUM_LAYERS}")
    print(f"Dropout             : " f"{DROPOUT}")
    print(f"Maximum sequence    : " f"{MAX_LENGTH}")
    print(f"Batch size          : " f"{BATCH_SIZE}")
    print(f"Epochs              : " f"{EPOCHS}")
    print(f"Learning rate       : " f"{LEARNING_RATE}")
    print(f"\nTotal parameters  : " f"{total_parameters:,}")
    print(f"Trainable parameters: " f"{trainable_parameters:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )
    best_val_f1 = 0.0
    best_epoch = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
            )
        )

        (
            val_loss,
            val_acc,
            val_macro_f1,
            val_weighted_f1,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
        )
        scheduler.step(val_macro_f1)
        current_lr = (optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_weighted_f1"].append(val_weighted_f1)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = (val_macro_f1)
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "config": {
                        "embedding_dim": EMBED_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "max_length": MAX_LENGTH,
                    },
                },
                MODEL_PATH,
            )

            print(
                f"  -> Best model saved "
                f"(Val Macro F1: "
                f"{best_val_f1:.4f})"
            )

    HISTORY_PATH = ("results/training_history.json")
    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
        )

    print(f"\nTraining history saved to: " f"{HISTORY_PATH}")
    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best epoch: " f"{best_epoch}")
    print(f"Best validation Macro F1: " f"{best_val_f1:.4f}")
    (
        val_loss,
        val_accuracy,
        val_macro_f1,
        val_weighted_f1,
        _,
        _,
    ) = evaluate(
        model,
        val_loader,
        criterion,
    )
    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)
    print(f"Accuracy    : " f"{val_accuracy:.4f}")
    print(f"Macro F1    : " f"{val_macro_f1:.4f}")
    print(f"Weighted F1 : " f"{val_weighted_f1:.4f}")
    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        test_weighted_f1,
        test_labels,
        test_predictions,
    ) = evaluate(
        model,
        test_loader,
        criterion,
    )
    print("\n" + "=" * 70)
    print("FINAL TEST")
    print("=" * 70)
    print(f"Accuracy    : " f"{test_accuracy:.4f}")
    print(f"Macro F1    : " f"{test_macro_f1:.4f}")
    print(f"Weighted F1 : " f"{test_weighted_f1:.4f}")
    target_names = [
        id_to_label[i]
        for i in range(num_classes)
    ]
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=list(range(num_classes)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best model: " f"{MODEL_PATH}")
    print(f"Training history: " f"{HISTORY_PATH}")

if __name__ == "__main__":
    main()