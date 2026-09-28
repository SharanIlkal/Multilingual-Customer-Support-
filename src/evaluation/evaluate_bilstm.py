import json
from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models.bilstm import BiLSTMClassifier
from src.training.bilstm_dataset import CustomerSupportDataset
from src.training.bilstm_tokenizer import SimpleTokenizer


ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data" / "final" / "multilingual" / "test.csv"

MODEL_DIR = ROOT / "models" / "bilstm"

MODEL_PATH = MODEL_DIR / "best_model.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
LABEL_PATH = MODEL_DIR / "label_encoder.json"

EXPECTED_LANGUAGES = {"en", "hi", "kn", "mr", "ta", "te"}


def load_test_data():

    test_df = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    actual_languages = set(
        test_df["language"].unique()
    )

    if actual_languages != EXPECTED_LANGUAGES:

        raise ValueError(
            "Test dataset languages do not match expected languages.\n"
            f"Expected: {EXPECTED_LANGUAGES}\n"
            f"Found: {actual_languages}"
        )

    return test_df


def load_model():

    with open(
        LABEL_PATH,
        encoding="utf-8",
    ) as file:

        mapping = json.load(file)

    label2idx = mapping["label2idx"]

    idx2label = {
        int(key): value
        for key, value in mapping["idx2label"].items()
    }

    tokenizer = SimpleTokenizer.load(
        VOCAB_PATH
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model = BiLSTMClassifier(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        padding_idx=0,
        dropout=0.3,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return (
        model,
        tokenizer,
        label2idx,
        idx2label,
        checkpoint,
    )


def evaluate_model(
    model,
    tokenizer,
    label2idx,
    checkpoint,
    test_df,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    dataset = CustomerSupportDataset(
        test_df,
        tokenizer,
        label2idx,
        checkpoint["max_length"],
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():

        for input_ids, labels in loader:

            input_ids = input_ids.to(device)

            logits, _ = model(
                input_ids
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            y_true.extend(
                labels.tolist()
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

    return y_true, y_pred


def save_predictions(
    test_df,
    y_true,
    y_pred,
    idx2label,
):

    output = test_df.copy()

    output["true_label_id"] = y_true

    output["predicted_label_id"] = y_pred

    output["predicted_intent"] = [
        idx2label[index]
        for index in y_pred
    ]

    output["correct"] = (
        output["intent"]
        == output["predicted_intent"]
    )

    output_path = (
        MODEL_DIR
        / "test_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nPredictions saved to:\n"
        f"{output_path}"
    )


def save_confusion_matrix(
    y_true,
    y_pred,
    idx2label,
):

    labels = list(
        range(len(idx2label))
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    intent_names = [
        idx2label[index]
        for index in labels
    ]

    matrix_df = pd.DataFrame(
        matrix,
        index=intent_names,
        columns=intent_names,
    )

    output_path = (
        MODEL_DIR
        / "confusion_matrix.csv"
    )

    matrix_df.to_csv(
        output_path,
        encoding="utf-8-sig",
    )

    print(
        f"Confusion matrix saved to:\n"
        f"{output_path}"
    )


def save_classification_report(
    y_true,
    y_pred,
    idx2label,
):

    labels = list(
        range(len(idx2label))
    )

    target_names = [
        idx2label[index]
        for index in labels
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    output_path = (
        MODEL_DIR
        / "classification_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "MULTILINGUAL BiLSTM + ATTENTION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(report)

    print(
        f"Classification report saved to:\n"
        f"{output_path}"
    )


def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MULTILINGUAL BiLSTM + ATTENTION EVALUATION")
    print("=" * 70)

    test_df = load_test_data()

    print(
        f"\nTest samples : {len(test_df)}"
    )

    print(
        f"Languages    : "
        f"{sorted(EXPECTED_LANGUAGES)}"
    )

    model, tokenizer, label2idx, idx2label, checkpoint = (
        load_model()
    )

    print(
        f"Classes      : {len(idx2label)}"
    )

    print(
        f"Vocabulary   : {checkpoint['vocab_size']}"
    )

    print(
        f"Max length   : {checkpoint['max_length']}"
    )

    print("\nRunning inference...")

    y_true, y_pred = evaluate_model(
        model,
        tokenizer,
        label2idx,
        checkpoint,
        test_df,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    print("\n")
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy    : {accuracy:.4f}"
    )

    print(
        f"Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(idx2label.keys()),
            target_names=[
                idx2label[index]
                for index in idx2label.keys()
            ],
            digits=4,
            zero_division=0,
        )
    )

    save_predictions(
        test_df,
        y_true,
        y_pred,
        idx2label,
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        idx2label,
    )

    save_classification_report(
        y_true,
        y_pred,
        idx2label,
    )

    print("\n")
    print("=" * 70)
    print("MULTILINGUAL BiLSTM SUMMARY")
    print("=" * 70)

    print(
        f"\nTest Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Test Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Test Weighted F1 : {weighted_f1:.4f}"
    )

    print(
        f"Test Samples     : {len(test_df)}"
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()