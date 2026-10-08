import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModel, AutoTokenizer

SEED = 42
TRAIN_PATH = "data/processed/final/train.csv"
VAL_PATH = "data/processed/final/validation.csv"
TEST_PATH = "data/processed/final/test.csv"
MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 64
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_label_mapping(train_df):
    labels = sorted(train_df["intent"].unique())
    label_to_id = {
        label: idx
        for idx, label in enumerate(labels)
    }
    id_to_label = {
        idx: label
        for label, idx in label_to_id.items()
    }
    return label_to_id, id_to_label

@torch.no_grad()
def extract_embeddings(
    texts,
    tokenizer,
    model,
    batch_size=BATCH_SIZE,
):
    all_embeddings = []
    print(f"Extracting embeddings for " f"{len(texts):,} samples...")
    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        end = min(start + batch_size, len(texts),)
        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }
        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        masked_embeddings = (token_embeddings* attention_mask)
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = (summed / counts)
        all_embeddings.append(embeddings.cpu().numpy())
        if (
            start == 0
            or end == len(texts)
            or (start // batch_size) % 100 == 0
        ):
            print(f"  Processed " f"{end:,}/{len(texts):,}")
    return np.vstack(all_embeddings)

def evaluate_classifier(
    classifier,
    embeddings,
    labels,
    id_to_label,
    split_name,
):

    predictions = classifier.predict(embeddings)
    accuracy = accuracy_score(labels,predictions,)
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print(split_name)
    print("=" * 70)
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    return (
        accuracy,
        macro_f1,
        weighted_f1,
        predictions,
    )

def main():
    set_seed()
    print("=" * 70)
    print("XLM-R EMBEDDINGS + LOGISTIC REGRESSION")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Model : {MODEL_NAME}")
    print("\nLoading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(val_df):,}")
    print(f"Test       : {len(test_df):,}")
    print("\nLanguages:")
    print(
        train_df["language"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    label_to_id, id_to_label = (build_label_mapping(train_df))
    print(f"\nNumber of classes: " f"{len(label_to_id)}")
    print("\nLoading tokenizer...")
    tokenizer = (AutoTokenizer.from_pretrained(MODEL_NAME))
    print("\nLoading XLM-R encoder...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Encoder parameters: " f"{total_parameters:,}")
    print("\nXLM-R encoder is frozen.")
    train_texts = (
        train_df["text"]
        .astype(str)
        .tolist()
    )

    val_texts = (
        val_df["text"]
        .astype(str)
        .tolist()
    )

    test_texts = (
        test_df["text"]
        .astype(str)
        .tolist()
    )

    train_labels = np.array(
        [
            label_to_id[label]
            for label in train_df["intent"]
        ]
    )

    val_labels = np.array(
        [
            label_to_id[label]
            for label in val_df["intent"]
        ]
    )

    test_labels = np.array(
        [
            label_to_id[label]
            for label in test_df["intent"]
        ]
    )
    print("\n" + "=" * 70)
    print("TRAIN EMBEDDINGS")
    print("=" * 70)
    X_train = extract_embeddings(
        train_texts,
        tokenizer,
        model,
    )
    print(f"Train embedding shape: " f"{X_train.shape}")
    print("\n" + "=" * 70)
    print("VALIDATION EMBEDDINGS")
    print("=" * 70)
    X_val = extract_embeddings(
        val_texts,
        tokenizer,
        model,
    )
    print(f"Validation embedding shape: " f"{X_val.shape}")
    print("\n" + "=" * 70)
    print("TEST EMBEDDINGS")
    print("=" * 70)
    X_test = extract_embeddings(
        test_texts,
        tokenizer,
        model,
    )
    print(f"Test embedding shape: " f"{X_test.shape}")
    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)
    classifier = LogisticRegression(
        max_iter=1000,
        C=4.0,
        solver="lbfgs",
    )
    classifier.fit(X_train, train_labels,)
    print("Classifier training complete.")
    (
        val_accuracy,
        val_macro_f1,
        val_weighted_f1,
        _,
    ) = evaluate_classifier(
        classifier,
        X_val,
        val_labels,
        id_to_label,
        "VALIDATION",
    )
    (
        test_accuracy,
        test_macro_f1,
        test_weighted_f1,
        test_predictions,
    ) = evaluate_classifier(
        classifier,
        X_test,
        test_labels,
        id_to_label,
        "TEST",
    )
    target_names = [
        id_to_label[i]
        for i in range(len(id_to_label))
    ]

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=list(range(len(id_to_label))),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    os.makedirs("results", exist_ok=True,)
    results = {
        "model": "XLM-R embeddings + Logistic Regression",
        "encoder": MODEL_NAME,
        "embedding_dimension": int(X_train.shape[1]),
        "validation": {
            "accuracy": float(val_accuracy),
            "macro_f1": float(val_macro_f1),
            "weighted_f1": float(val_weighted_f1),
        },
        "test": {
            "accuracy": float(test_accuracy),
            "macro_f1": float(test_macro_f1),
            "weighted_f1": float(test_weighted_f1),
        },
    }

    import json
    with open(
        "results/xlmr_embeddings_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(results, f,indent=4,)
    print(
        "\nResults saved to:"
        "\nresults/xlmr_embeddings_results.json"
    )

    print("\n" + "=" * 70)
    print("XLM-R EMBEDDING CLASSIFIER COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()