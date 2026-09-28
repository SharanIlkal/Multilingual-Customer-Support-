from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data"
    / "final"
    / "multilingual"
)

TRAIN_PATH = (
    DATA_DIR / "train.csv"
)

VALIDATION_PATH = (
    DATA_DIR / "validation.csv"
)

TEST_PATH = (
    DATA_DIR / "test.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "tfidf"
)

MODEL_PATH = (
    MODEL_DIR
    / "tfidf_logistic_regression.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH,
        encoding="utf-8-sig",
    )

    test_df = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100000,
        strip_accents=None,
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                vectorizer,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    x,
    y,
    dataset_name,
):

    predictions = model.predict(x)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
    )

    print("\n")
    print("=" * 70)
    print(
        f"{dataset_name.upper()} RESULTS"
    )
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
            y,
            predictions,
            digits=4,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": predictions,
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test_df,
    predictions,
):

    prediction_df = test_df.copy()

    prediction_df[
        "predicted_intent"
    ] = predictions

    prediction_df[
        "correct"
    ] = (
        prediction_df["intent"]
        == prediction_df["predicted_intent"]
    )

    output_path = (
        MODEL_DIR
        / "test_predictions.csv"
    )

    prediction_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nPredictions saved to:\n"
        f"{output_path}"
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    model,
    x_test,
    y_test,
):

    predictions = model.predict(
        x_test
    )

    labels = sorted(
        y_test.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
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


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_test,
    predictions,
):

    report = classification_report(
        y_test,
        predictions,
        digits=4,
    )

    output_path = (
        MODEL_DIR
        / "classification_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TF-IDF + LOGISTIC REGRESSION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(report)

    print(
        f"Classification report saved to:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "MULTILINGUAL TF-IDF + "
        "LOGISTIC REGRESSION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = load_data()

    print(
        f"\nTraining samples   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_df)}"
    )

    print(
        f"Test samples       : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Validate languages
    # --------------------------------------------------------

    expected_languages = {
        "en",
        "hi",
        "kn",
        "mr",
        "ta",
        "te",
    }

    for name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:

        actual_languages = set(
            df["language"].unique()
        )

        if actual_languages != expected_languages:

            raise ValueError(
                f"{name} dataset languages do not match "
                f"expected languages.\n"
                f"Expected: {expected_languages}\n"
                f"Found: {actual_languages}"
            )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    x_train = train_df["text"]
    y_train = train_df["intent"]

    x_validation = validation_df["text"]
    y_validation = validation_df["intent"]

    x_test = test_df["text"]
    y_test = test_df["intent"]

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining multilingual "
        "TF-IDF model..."
    )

    model.fit(
        x_train,
        y_train,
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    evaluate_model(
        model,
        x_validation,
        y_validation,
        "validation",
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        x_test,
        y_test,
        "test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_predictions(
        test_df,
        test_results["predictions"],
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        model,
        x_test,
        y_test,
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    save_classification_report(
        y_test,
        test_results["predictions"],
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "MULTILINGUAL TF-IDF SUMMARY"
    )
    print("=" * 70)

    print(
        f"\nTest Accuracy    : "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Test Macro F1    : "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Test Weighted F1 : "
        f"{test_results['weighted_f1']:.4f}"
    )

    print(
        f"Test Samples     : "
        f"{len(test_df)}"
    )

    print(
        "\nMultilingual TF-IDF model completed."
    )


if __name__ == "__main__":
    main()