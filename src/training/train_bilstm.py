import json
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.models.bilstm import BiLSTMClassifier
from src.training.bilstm_tokenizer import SimpleTokenizer
from src.training.bilstm_dataset import CustomerSupportDataset

SEED = 42
TRAIN_PATH = "data/final/multilingual/train.csv"
VAL_PATH = "data/final/multilingual/validation.csv"
MODEL_DIR = "models/bilstm"
MAX_LENGTH = 64
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 15
PATIENCE = 3
EXPECTED_LANGUAGES = {"en", "hi", "kn", "mr", "ta", "te"}

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits, _ = model(input_ids)
            loss = criterion(logits, labels,)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1,)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    average_loss = total_loss / len(dataloader)
    return average_loss, accuracy

def validate_languages(df, name):
    actual_languages = set(df["language"].dropna().unique())
    if actual_languages != EXPECTED_LANGUAGES:
        raise ValueError(
            f"{name} dataset languages do not match expected languages.\n"
            f"Expected: {EXPECTED_LANGUAGES}\n"
            f"Found: {actual_languages}"
        )

def main():
    set_seed()
    os.makedirs(MODEL_DIR, exist_ok=True,)
    print("=" * 70)
    print("MULTILINGUAL BiLSTM + ATTENTION TRAINING")
    print("=" * 70)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig",)
    val_df = pd.read_csv(VAL_PATH, encoding="utf-8-sig",)
    print(f"\nTraining samples   : {len(train_df)}")
    print(f"Validation samples : {len(val_df)}")
    validate_languages(train_df,"Training",)
    validate_languages(val_df,"Validation",)
    print(f"Languages          : "f"{sorted(EXPECTED_LANGUAGES)}")
    intents = sorted(train_df["intent"].unique())
    label2idx = {
        label: index
        for index, label in enumerate(intents)
    }
    idx2label = {
        index: label
        for label, index in label2idx.items()
    }
    num_classes = len(intents)
    print(f"Number of classes  : {num_classes}")
    print("\nBuilding multilingual vocabulary...")
    tokenizer = SimpleTokenizer(
        min_freq=2,
        max_vocab_size=30000,
    )
    tokenizer.build_vocab(train_df["text"].tolist())
    vocab_size = len(tokenizer.word2idx)
    print(f"Vocabulary size    : {vocab_size}")
    tokenizer.save(f"{MODEL_DIR}/vocab.json")
    with open(
        f"{MODEL_DIR}/label_encoder.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "label2idx": label2idx,
                "idx2label": idx2label,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    train_dataset = CustomerSupportDataset(
        train_df,
        tokenizer,
        label2idx,
        MAX_LENGTH,
    )

    val_dataset = CustomerSupportDataset(
        val_df,
        tokenizer,
        label2idx,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        padding_idx=0,
        dropout=0.3,
    )
    model = model.to(device)
    print("\nModel:")
    print(model)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: " f"{total_parameters:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,)
    best_val_accuracy = 0.0
    patience_counter = 0
    history = []
    print("\nStarting training...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits,labels,)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = torch.argmax(logits,dim=1,)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        train_loss = (running_loss/ len(train_loader))
        train_accuracy = (correct / total)
        val_loss, val_accuracy = evaluate(model, val_loader,device,)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "vocab_size":
                        vocab_size,
                    "embedding_dim":
                        EMBEDDING_DIM,
                    "hidden_dim":
                        HIDDEN_DIM,
                    "num_classes":
                        num_classes,
                    "max_length":
                        MAX_LENGTH,
                },
                f"{MODEL_DIR}/best_model.pt",
            )
            print("Best model saved.")
        else:
            patience_counter += 1
            print(f"  No improvement " f"({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break
    with open(
        f"{MODEL_DIR}/training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file,indent=2,)
    print("\n" + "=" * 70)
    print("MULTILINGUAL TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nBest validation accuracy: " f"{best_val_accuracy:.4f}")
    print(f"\nModel saved to:" f"\n{MODEL_DIR}/best_model.pt")
    print(f"\nVocabulary saved to:" f"\n{MODEL_DIR}/vocab.json")
    print(f"\nLabel mapping saved to:" f"\n{MODEL_DIR}/label_encoder.json")

if __name__ == "__main__":
    main()