from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/intermediate/bitext_normalized.csv"
)

OUTPUT_DIR = Path(
    "data/final"
)

RANDOM_STATE = 42

# 80% train
# 10% validation
# 10% test

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    path: str | Path,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    print(
        f"Loaded dataset: {len(df)} samples"
    )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
):

    required_columns = {
        "text",
        "language",
        "intent",
        "category",
        "source",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # Check missing values
    required_data = df[
        [
            "text",
            "intent",
        ]
    ]

    if required_data.isnull().any().any():

        raise ValueError(
            "Dataset contains missing values "
            "in text or intent."
        )

    # Check minimum samples per intent
    intent_counts = (
        df["intent"]
        .value_counts()
    )

    print("\nIntent sample counts:")

    print(
        intent_counts
        .sort_values()
        .to_string()
    )

    if intent_counts.min() < 3:

        raise ValueError(
            "At least 3 samples are required "
            "for every intent."
        )


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(
    df: pd.DataFrame,
):

    # --------------------------------------------------------
    # First split:
    #
    # 80% train
    # 20% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(
            VALIDATION_SIZE + TEST_SIZE
        ),
        random_state=RANDOM_STATE,
        stratify=df["intent"],
    )

    # --------------------------------------------------------
    # Second split:
    #
    # temporary 20%
    #
    # → 10% validation
    # → 10% test
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["intent"],
    )

    # Reset indexes
    train_df = train_df.reset_index(
        drop=True
    )

    validation_df = validation_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# CHECK FOR DATA LEAKAGE
# ============================================================

def check_data_leakage(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    train_texts = set(
        train_df["text"]
    )

    validation_texts = set(
        validation_df["text"]
    )

    test_texts = set(
        test_df["text"]
    )

    validation_overlap = (
        train_texts
        & validation_texts
    )

    test_overlap = (
        train_texts
        & test_texts
    )

    validation_test_overlap = (
        validation_texts
        & test_texts
    )

    print("\n" + "=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)

    print(
        f"\nTrain ↔ Validation overlap: "
        f"{len(validation_overlap)}"
    )

    print(
        f"Train ↔ Test overlap: "
        f"{len(test_overlap)}"
    )

    print(
        f"Validation ↔ Test overlap: "
        f"{len(validation_test_overlap)}"
    )

    if (
        validation_overlap
        or test_overlap
        or validation_test_overlap
    ):

        raise ValueError(
            "Data leakage detected!"
        )

    print("\nNo exact text leakage detected.")


# ============================================================
# PRINT SPLIT STATISTICS
# ============================================================

def print_split_statistics(
    name: str,
    df: pd.DataFrame,
):

    print("\n" + "-" * 70)

    print(
        f"{name.upper()} DATASET"
    )

    print("-" * 70)

    print(
        f"Samples   : {len(df)}"
    )

    print(
        f"Intents   : {df['intent'].nunique()}"
    )

    print(
        f"Categories: {df['category'].nunique()}"
    )

    print("\nIntent distribution:")

    print(
        df["intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# SAVE DATASETS
# ============================================================

def save_dataset(
    df: pd.DataFrame,
    filename: str,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / filename
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CUSTOMER SUPPORT DATASET SPLITTER")
    print("=" * 70)

    # Load
    df = load_dataset(
        INPUT_PATH
    )

    # Validate
    validate_dataset(df)

    # Split
    (
        train_df,
        validation_df,
        test_df,
    ) = split_dataset(df)

    # Check leakage
    check_data_leakage(
        train_df,
        validation_df,
        test_df,
    )

    # Statistics
    print_split_statistics(
        "train",
        train_df,
    )

    print_split_statistics(
        "validation",
        validation_df,
    )

    print_split_statistics(
        "test",
        test_df,
    )

    # Save
    save_dataset(
        train_df,
        "train.csv",
    )

    save_dataset(
        validation_df,
        "validation.csv",
    )

    save_dataset(
        test_df,
        "test.csv",
    )

    print("\n" + "=" * 70)
    print("DATASET SPLIT COMPLETED")
    print("=" * 70)

    print(
        f"\nTotal      : {len(df)}"
    )

    print(
        f"Train      : {len(train_df)} "
        f"({len(train_df) / len(df):.1%})"
    )

    print(
        f"Validation : {len(validation_df)} "
        f"({len(validation_df) / len(df):.1%})"
    )

    print(
        f"Test       : {len(test_df)} "
        f"({len(test_df) / len(df):.1%})"
    )


if __name__ == "__main__":
    main()