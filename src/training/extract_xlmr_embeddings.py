from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final" / "multilingual"
OUTPUT_DIR = ROOT / "data" / "intermediate" / "xlmr_embeddings"

TRAIN_PATH = DATA_DIR / "train_augmented.csv"
VAL_PATH = DATA_DIR / "validation.csv"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 64
BATCH_SIZE = 16


def mean_pooling(
    last_hidden_state,
    attention_mask
):
    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


def extract_embeddings(
    model,
    tokenizer,
    texts,
    device,
    split_name
):
    embeddings = []

    print(
        f"\nExtracting embeddings for "
        f"{len(texts)} {split_name} samples..."
    )

    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            BATCH_SIZE
        ):
            batch_texts = texts[
                start:start + BATCH_SIZE
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )

            batch_embeddings = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

            processed = min(
                start + BATCH_SIZE,
                len(texts)
            )

            print(
                f"Processed "
                f"{processed}/{len(texts)}"
            )

    return np.concatenate(
        embeddings,
        axis=0
    )


def process_split(
    path,
    split_name,
    model,
    tokenizer,
    device
):
    print(
        f"\nLoading {split_name} dataset..."
    )

    df = pd.read_csv(path)

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = (
        df["intent"]
        .astype(str)
        .tolist()
    )

    print(
        f"{split_name.capitalize()} samples   : "
        f"{len(df)}"
    )

    print(
        f"Unique intents       : "
        f"{df['intent'].nunique()}"
    )

    print(
        f"Languages            : "
        f"{', '.join(sorted(df['language'].unique()))}"
    )

    embeddings = extract_embeddings(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        split_name=split_name
    )

    embedding_path = (
        OUTPUT_DIR /
        f"{split_name}_embeddings.npy"
    )

    labels_path = (
        OUTPUT_DIR /
        f"{split_name}_labels.npy"
    )

    np.save(
        embedding_path,
        embeddings
    )

    np.save(
        labels_path,
        np.array(labels)
    )

    print(
        f"\nSaved {split_name} embeddings:"
    )

    print(embedding_path)

    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        f"Saved {split_name} labels:"
    )

    print(labels_path)

    print(
        f"Labels shape: {len(labels)}"
    )


def main():

    print("=" * 70)
    print("XLM-R MULTILINGUAL EMBEDDING EXTRACTION")
    print("=" * 70)

    print(
        f"\nTrain dataset:"
        f"\n{TRAIN_PATH}"
    )

    print(
        f"\nValidation dataset:"
        f"\n{VAL_PATH}"
    )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{TRAIN_PATH}"
        )

    if not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n"
            f"{VAL_PATH}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------
    # Verify datasets before loading XLM-R
    # --------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    val_df = pd.read_csv(
        VAL_PATH
    )

    print("\nLoading datasets...")

    print(
        f"Training samples   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_df)}"
    )

    expected_train = len(train_df)
    expected_val = 5123

    if len(train_df) != expected_train:
        raise ValueError(
            f"Expected multilingual training "
            f"dataset with {expected_train} samples, "
            f"but found {len(train_df)}."
        )

    if len(val_df) != expected_val:
        raise ValueError(
            f"Expected multilingual validation "
            f"dataset with {expected_val} samples, "
            f"but found {len(val_df)}."
        )

    expected_languages = {
        "en",
        "hi",
        "kn",
        "mr",
        "ta",
        "te"
    }

    train_languages = set(
        train_df["language"].unique()
    )

    val_languages = set(
        val_df["language"].unique()
    )

    if train_languages != expected_languages:
        raise ValueError(
            "Training dataset language mismatch.\n"
            f"Expected: {sorted(expected_languages)}\n"
            f"Found: {sorted(train_languages)}"
        )

    if val_languages != expected_languages:
        raise ValueError(
            "Validation dataset language mismatch.\n"
            f"Expected: {sorted(expected_languages)}\n"
            f"Found: {sorted(val_languages)}"
        )

    print(
        "\nDataset verification passed."
    )

    print(
        f"Training languages: "
        f"{sorted(train_languages)}"
    )

    print(
        f"Validation languages: "
        f"{sorted(val_languages)}"
    )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------
    # Load XLM-R
    # --------------------------------------------------

    print(
        "\nLoading XLM-R encoder..."
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print(
        "\nXLM-R loaded successfully."
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Extract TRAIN embeddings
    # --------------------------------------------------

    process_split(
        path=TRAIN_PATH,
        split_name="train",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # --------------------------------------------------
    # Extract VALIDATION embeddings
    # --------------------------------------------------

    process_split(
        path=VAL_PATH,
        split_name="validation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTILINGUAL EMBEDDING EXTRACTION COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()