import json
import os
import random

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
)

from src.models.xlmr_classifier import (
    XLMRClassifier,
)

from src.training.xlmr_dataset import (
    XLMRDataset,
)


SEED = 42

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TRAIN_PATH = "data/final/train.csv"
VAL_PATH = "data/final/validation.csv"

MODEL_DIR = "models/xlmr"

MAX_LENGTH = 128
BATCH_SIZE = 8

LEARNING_RATE = 2e-4

EPOCHS = 5

PATIENCE = 2


def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)


def evaluate(
    model,
    dataloader,
    device,
):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            logits = model(
                input_ids,
                attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / len(dataloader),
        correct / total,
    )


def main():

    set_seed()

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    print("=" * 70)
    print("XLM-RoBERTa CLASSIFIER")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    val_df = pd.read_csv(
        VAL_PATH
    )

    print(
        f"\nTraining samples   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_df)}"
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    intents = sorted(
        train_df["intent"].unique()
    )

    label2idx = {
        label: index
        for index, label in enumerate(
            intents
        )
    }

    idx2label = {
        index: label
        for label, index in label2idx.items()
    }

    num_classes = len(intents)

    print(
        f"Number of classes  : "
        f"{num_classes}"
    )

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
            indent=2,
        )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print("\nLoading XLM-R tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = XLMRDataset(
        train_df,
        tokenizer,
        label2idx,
        MAX_LENGTH,
    )

    val_dataset = XLMRDataset(
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

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nLoading XLM-R model...")

    model = XLMRClassifier(
        model_name=MODEL_NAME,
        num_classes=num_classes,
        dropout=0.3,
        freeze_encoder=True,
    )

    model = model.to(device)

    # --------------------------------------------------
    # Trainable parameters
    # --------------------------------------------------

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_val_accuracy = 0.0

    patience_counter = 0

    history = []

    print("\nStarting training...\n")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for batch in train_loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                input_ids,
                attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss
            / len(train_loader)
        )

        train_accuracy = (
            correct / total
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

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

            best_val_accuracy = (
                val_accuracy
            )

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "model_name":
                        MODEL_NAME,
                    "num_classes":
                        num_classes,
                    "max_length":
                        MAX_LENGTH,
                },
                f"{MODEL_DIR}/best_model.pt",
            )

            print(
                "  ✓ Best model saved."
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------
    # Save history
    # --------------------------------------------------

    with open(
        f"{MODEL_DIR}/training_history.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("XLM-R TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_DIR}/best_model.pt"
    )


if __name__ == "__main__":
    main()