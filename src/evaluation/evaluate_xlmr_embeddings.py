import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

from src.models.xlmr_embedding_classifier import (
    XLMREmbeddingClassifier,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data"
    / "final"
    / "multilingual"
)

TEST_PATH = DATA_DIR / "test.csv"

TEST_EMBEDDINGS = (
    ROOT
    / "data"
    / "intermediate"
    / "xlmr_embeddings"
    / "test_embeddings.npy"
)

TEST_LABELS = (
    ROOT
    / "data"
    / "intermediate"
    / "xlmr_embeddings"
    / "test_labels.npy"
)

CHECKPOINT = (
    ROOT
    / "models"
    / "xlmr"
    / "embedding_classifier.pt"
)

LABEL_ENCODER = (
    ROOT
    / "models"
    / "xlmr"
    / "embedding_label_encoder.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "xlmr"
)

EXPECTED_LANGUAGES = [
    "en",
    "hi",
    "kn",
    "mr",
    "ta",
    "te",
]


def load_label_mapping():

    with open(
        LABEL_ENCODER,
        "r",
        encoding="utf-8",
    ) as file:

        mapping = json.load(file)

    if "idx2label" not in mapping:

        raise ValueError(
            "embedding_label_encoder.json "
            "does not contain 'idx2label'."
        )

    return {
        int(key): value
        for key, value in mapping["idx2label"].items()
    }


def load_test_data():

    test_df = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    required_columns = [
        "text",
        "language",
        "intent",
    ]

    for column in required_columns:

        if column not in test_df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    actual_languages = sorted(
        test_df["language"].unique()
    )

    if set(actual_languages) != set(
        EXPECTED_LANGUAGES
    ):

        raise ValueError(
            "Unexpected languages found.\n"
            f"Expected: {EXPECTED_LANGUAGES}\n"
            f"Found: {actual_languages}"
        )

    return test_df


def load_test_embeddings():

    X_test = np.load(
        TEST_EMBEDDINGS
    )

    y_test = np.load(
        TEST_LABELS,
        allow_pickle=True,
    )

    if len(X_test) != len(y_test):

        raise ValueError(
            "Test embeddings and labels "
            "have different lengths."
        )

    if X_test.shape[1] != 768:

        raise ValueError(
            f"Expected 768-dimensional embeddings, "
            f"got {X_test.shape[1]}."
        )

    return X_test, y_test


def convert_labels(
    y_test,
    id_to_label,
):

    label_to_id = {
        label: idx
        for idx, label in id_to_label.items()
    }

    if y_test.dtype.kind in {
        "U",
        "S",
        "O",
    }:

        unknown_labels = sorted(
            set(y_test.tolist())
            - set(label_to_id.keys())
        )

        if unknown_labels:

            raise ValueError(
                f"Unknown labels found: "
                f"{unknown_labels}"
            )

        return np.array(
            [
                label_to_id[label]
                for label in y_test
            ],
            dtype=np.int64,
        )

    return y_test.astype(
        np.int64
    )


def load_model(
    num_classes,
):

    device = torch.device("cpu")

    model = XLMREmbeddingClassifier(
        input_dim=768,
        num_classes=num_classes,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    return model, device


def predict(
    model,
    device,
    X_test,
):

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )

    return predictions


def save_predictions(
    test_df,
    y_test,
    predictions,
    id_to_label,
):

    output = test_df.copy()

    output["true_intent"] = [
        id_to_label[int(x)]
        for x in y_test
    ]

    output["predicted_intent"] = [
        id_to_label[int(x)]
        for x in predictions
    ]

    output["correct"] = (
        y_test == predictions
    )

    output_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nPredictions saved:\n"
        f"{output_path}"
    )


def save_per_language_results(
    test_df,
    y_test,
    predictions,
    id_to_label,
):

    rows = []

    for language in EXPECTED_LANGUAGES:

        indices = (
            test_df["language"]
            .values
            == language
        )

        language_true = y_test[indices]
        language_pred = predictions[indices]

        accuracy = accuracy_score(
            language_true,
            language_pred,
        )

        macro_f1 = f1_score(
            language_true,
            language_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            language_true,
            language_pred,
            average="weighted",
            zero_division=0,
        )

        rows.append(
            {
                "language": language,
                "samples": len(language_true),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

    results = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIR
        / "per_language_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("PER-LANGUAGE RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\nPer-language results saved:\n"
        f"{output_path}"
    )

    return results


def save_classification_report(
    y_test,
    predictions,
    class_names,
):

    report = classification_report(
        y_test,
        predictions,
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    output_path = (
        OUTPUT_DIR
        / "classification_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "MULTILINGUAL XLM-R "
            "EMBEDDING CLASSIFIER\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "Classification Report:\n\n"
        )

        file.write(report)

    print(
        f"\nClassification report saved:\n"
        f"{output_path}"
    )

    return report


def save_confusion_matrix(
    y_test,
    predictions,
    class_names,
):

    labels = list(
        range(len(class_names))
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    cm_path = (
        OUTPUT_DIR
        / "confusion_matrix.npy"
    )

    np.save(
        cm_path,
        cm,
    )

    print(
        f"\nConfusion matrix saved:\n"
        f"{cm_path}"
    )

    fig, ax = plt.subplots(
        figsize=(18, 16)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
        colorbar=True,
    )

    ax.set_title(
        "Multilingual XLM-R Embedding "
        "Classifier - Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Intent"
    )

    ax.set_ylabel(
        "True Intent"
    )

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / "multilingual_confusion_matrix.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nConfusion matrix plot saved:\n"
        f"{plot_path}"
    )

    return cm


def print_top_confusions(
    cm,
    class_names,
):

    confusion_pairs = []

    for true_id in range(
        len(class_names)
    ):

        for pred_id in range(
            len(class_names)
        ):

            if true_id == pred_id:
                continue

            count = cm[
                true_id,
                pred_id
            ]

            if count > 0:

                confusion_pairs.append(
                    (
                        count,
                        class_names[true_id],
                        class_names[pred_id],
                    )
                )

    confusion_pairs.sort(
        reverse=True
    )

    print("\n" + "=" * 70)
    print("TOP CONFUSIONS")
    print("=" * 70)

    for (
        count,
        true_label,
        predicted_label,
    ) in confusion_pairs[:20]:

        print(
            f"{true_label:30s} -> "
            f"{predicted_label:30s} : {count}"
        )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MULTILINGUAL XLM-R EMBEDDING EVALUATION")
    print("=" * 70)

    test_df = load_test_data()

    X_test, y_test_raw = (
        load_test_embeddings()
    )

    id_to_label = (
        load_label_mapping()
    )

    class_names = [
        id_to_label[i]
        for i in sorted(id_to_label)
    ]

    y_test = convert_labels(
        y_test_raw,
        id_to_label,
    )

    print(
        f"\nTest samples          : {len(test_df)}"
    )

    print(
        f"Test embeddings shape : {X_test.shape}"
    )

    print(
        f"Number of classes     : "
        f"{len(class_names)}"
    )

    print(
        f"Languages             : "
        f"{EXPECTED_LANGUAGES}"
    )

    model, device = load_model(
        len(class_names)
    )

    print(
        f"\nDevice: {device}"
    )

    print("\nRunning inference...")

    predictions = predict(
        model,
        device,
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"\nTest Accuracy    : "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Macro F1    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Test Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    report = save_classification_report(
        y_test,
        predictions,
        class_names,
    )

    print(
        "\nClassification Report:\n"
    )

    print(report)

    save_predictions(
        test_df,
        y_test,
        predictions,
        id_to_label,
    )

    cm = save_confusion_matrix(
        y_test,
        predictions,
        class_names,
    )

    save_per_language_results(
        test_df,
        y_test,
        predictions,
        id_to_label,
    )

    print_top_confusions(
        cm,
        class_names,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()