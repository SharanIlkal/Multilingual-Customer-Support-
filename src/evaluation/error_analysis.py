import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    ROOT
    / "data"
    / "final"
    / "multilingual"
    / "test.csv"
)

PREDICTIONS_PATH = (
    ROOT
    / "models"
    / "bilstm"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "bilstm"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MULTILINGUAL BiLSTM ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    print(
        f"\nTest dataset samples: "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    predictions_df = pd.read_csv(
        PREDICTIONS_PATH,
        encoding="utf-8-sig",
    )

    print(
        f"Prediction samples: "
        f"{len(predictions_df)}"
    )

    # --------------------------------------------------------
    # Validate sizes
    # --------------------------------------------------------

    if len(test_df) != len(predictions_df):

        raise ValueError(
            "Test dataset and prediction sizes do not match.\n"
            f"Test: {len(test_df)}\n"
            f"Predictions: {len(predictions_df)}"
        )

    # --------------------------------------------------------
    # Determine prediction column
    # --------------------------------------------------------

    prediction_column = None

    for column in [
        "predicted_intent",
        "prediction",
        "predicted_label",
    ]:

        if column in predictions_df.columns:

            prediction_column = column
            break

    if prediction_column is None:

        raise ValueError(
            "Could not find prediction column."
        )

    # --------------------------------------------------------
    # Build analysis dataframe
    # --------------------------------------------------------

    analysis_df = test_df.copy()

    analysis_df["predicted_intent"] = (
        predictions_df[prediction_column].values
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "text",
        "language",
        "intent",
    ]

    for column in required_columns:

        if column not in analysis_df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # --------------------------------------------------------
    # Correct / incorrect
    # --------------------------------------------------------

    analysis_df["correct"] = (
        analysis_df["intent"]
        == analysis_df["predicted_intent"]
    )

    total = len(analysis_df)

    correct = int(
        analysis_df["correct"].sum()
    )

    incorrect = total - correct

    error_rate = incorrect / total

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"\nTotal: {total}"
        f" | Correct: {correct}"
        f" | Incorrect: {incorrect}"
        f" | Error rate: {error_rate:.4%}"
    )

    # ========================================================
    # ERRORS
    # ========================================================

    errors = analysis_df[
        ~analysis_df["correct"]
    ].copy()

    # ========================================================
    # ERRORS BY LANGUAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("ERRORS BY LANGUAGE")
    print("=" * 70)

    language_errors = (
        errors["language"]
        .value_counts()
        .sort_index()
    )

    print(language_errors)

    language_summary = (
        analysis_df
        .groupby("language")
        .agg(
            total=("correct", "size"),
            errors=(
                "correct",
                lambda x: (~x).sum(),
            ),
        )
    )

    language_summary["correct"] = (
        language_summary["total"]
        - language_summary["errors"]
    )

    language_summary["accuracy"] = (
        language_summary["correct"]
        / language_summary["total"]
    )

    language_summary["error_rate"] = (
        language_summary["errors"]
        / language_summary["total"]
    )

    language_summary = (
        language_summary
        .sort_values(
            "error_rate",
            ascending=False,
        )
    )

    print("\nLanguage performance:")
    print(language_summary)

    # ========================================================
    # ERRORS BY ACTUAL INTENT
    # ========================================================

    print("\n" + "=" * 70)
    print("ERRORS BY ACTUAL INTENT")
    print("=" * 70)

    errors_by_intent = (
        errors["intent"]
        .value_counts()
    )

    print(errors_by_intent)

    # ========================================================
    # CONFUSIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("CONFUSIONS")
    print("=" * 70)

    if not errors.empty:

        confusions = (
            errors
            .groupby(
                [
                    "intent",
                    "predicted_intent",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False,
            )
        )

        print(
            confusions.to_string(
                index=False
            )
        )

    # ========================================================
    # ERRORS BY LANGUAGE + INTENT
    # ========================================================

    print("\n" + "=" * 70)
    print("ERRORS BY LANGUAGE AND ACTUAL INTENT")
    print("=" * 70)

    if not errors.empty:

        language_intent_errors = (
            errors
            .groupby(
                [
                    "language",
                    "intent",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False,
            )
        )

        print(
            language_intent_errors.to_string(
                index=False
            )
        )

    # ========================================================
    # MISCLASSIFIED EXAMPLES
    # ========================================================

    print("\n" + "=" * 70)
    print("MISCLASSIFIED EXAMPLES")
    print("=" * 70)

    if errors.empty:

        print(
            "\nNo misclassified examples."
        )

    else:

        display_columns = [
            "language",
            "text",
            "intent",
            "predicted_intent",
        ]

        print(
            errors[
                display_columns
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE ERROR REPORT
    # ========================================================

    error_output_path = (
        OUTPUT_DIR
        / "multilingual_error_analysis.csv"
    )

    errors.to_csv(
        error_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved error analysis:"
        f"\n{error_output_path}"
    )

    # ========================================================
    # SAVE LANGUAGE SUMMARY
    # ========================================================

    language_summary_path = (
        OUTPUT_DIR
        / "language_error_summary.csv"
    )

    language_summary.to_csv(
        language_summary_path,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved language summary:"
        f"\n{language_summary_path}"
    )

    # ========================================================
    # SAVE CONFUSION REPORT
    # ========================================================

    if not errors.empty:

        confusion_output_path = (
            OUTPUT_DIR
            / "confusion_pairs.csv"
        )

        confusions.to_csv(
            confusion_output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nSaved confusion pairs:"
            f"\n{confusion_output_path}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("MULTILINGUAL BiLSTM ERROR ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nTotal samples : {total}"
    )

    print(
        f"Correct       : {correct}"
    )

    print(
        f"Incorrect     : {incorrect}"
    )

    print(
        f"Error rate    : {error_rate:.4%}"
    )


if __name__ == "__main__":
    main()