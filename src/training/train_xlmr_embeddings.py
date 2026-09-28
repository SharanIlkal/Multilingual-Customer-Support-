import json
import os
import random

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

from src.models.xlmr_embedding_classifier import (
    XLMREmbeddingClassifier,
)


SEED = 42

EMBEDDING_DIR = (
    "data/intermediate/xlmr_embeddings"
)

TRAIN_PATH = "data/final/multilingual/train_augmented.csv"
VAL_PATH = "data/final/multilingual/validation.csv"

MODEL_DIR = "models/xlmr"

BATCH_SIZE = 128

LEARNING_RATE = 0.001

EPOCHS = 30

PATIENCE = 5


def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for embeddings, labels in dataloader:

            embeddings = embeddings.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                embeddings
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

    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = correct / total

    return average_loss, accuracy


def main():

    set_seed()

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    print("=" * 70)
    print("XLM-R EMBEDDING CLASSIFIER TRAINING")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------
    # Load embeddings
    # --------------------------------------------------

    print("\nLoading embeddings...")

    X_train = np.load(
        f"{EMBEDDING_DIR}/train_embeddings.npy"
    )

    X_val = np.load(
        f"{EMBEDDING_DIR}/validation_embeddings.npy"
    )

    print(
        f"Train embeddings: "
        f"{X_train.shape}"
    )

    print(
        f"Validation embeddings: "
        f"{X_val.shape}"
    )

    # --------------------------------------------------
    # Load labels
    # --------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    val_df = pd.read_csv(
        VAL_PATH
    )

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
        f"Number of classes: "
        f"{num_classes}"
    )

    train_labels = np.array(
        [
            label2idx[label]
            for label in train_df["intent"]
        ],
        dtype=np.int64,
    )

    val_labels = np.array(
        [
            label2idx[label]
            for label in val_df["intent"]
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            train_labels,
            dtype=torch.long,
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            val_labels,
            dtype=torch.long,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = XLMREmbeddingClassifier(
        input_dim=768,
        hidden_dim=256,
        num_classes=num_classes,
        dropout=0.3,
    )

    model = model.to(device)

    print("\nModel:")
    print(model)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nClassifier parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_val_accuracy = 0.0

    patience_counter = 0

    history = []

    print("\nStarting classifier training...\n")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for embeddings, labels in train_loader:

            embeddings = embeddings.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                embeddings
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
            criterion,
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

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim":
                        768,

                    "hidden_dim":
                        256,

                    "num_classes":
                        num_classes,
                },
                f"{MODEL_DIR}/embedding_classifier.pt",
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
    # Save mappings/history
    # --------------------------------------------------

    with open(
        f"{MODEL_DIR}/embedding_label_encoder.json",
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

    with open(
        f"{MODEL_DIR}/embedding_training_history.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("XLM-R EMBEDDING CLASSIFIER COMPLETED")
    print("=" * 70)

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_DIR}/embedding_classifier.pt"
    )


if __name__ == "__main__":
    main()