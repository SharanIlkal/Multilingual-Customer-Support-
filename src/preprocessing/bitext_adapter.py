from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/raw/external_datasets/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

OUTPUT_PATH = Path(
    "data/intermediate/"
    "bitext_normalized.csv"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "instruction",
    "category",
    "intent",
    "response",
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_bitext_dataset(
    path: str | Path = INPUT_PATH,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Bitext dataset not found:\n{path}"
        )

    print(f"Loading dataset: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8",
    )

    print(f"Loaded rows: {len(df)}")

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df: pd.DataFrame):

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print("Dataset validation: PASSED")


# ============================================================
# NORMALIZE TEXT
# ============================================================

def clean_text(text: str) -> str:

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Normalize repeated whitespace
    text = " ".join(text.split())

    return text


# ============================================================
# NORMALIZE DATASET
# ============================================================

def normalize_bitext(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Customer message
    # --------------------------------------------------------

    df["text"] = df["instruction"].apply(
        clean_text
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    # Bitext dataset is English.
    df["language"] = "en"

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    # Bitext doesn't provide sentiment labels.
    # We will create this in a separate pipeline.
    df["sentiment"] = "unknown"

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    # Bitext doesn't provide priority labels.
    # We will create this separately.
    df["priority"] = "unknown"

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    df["source"] = "bitext"

    # --------------------------------------------------------
    # Select required columns
    # --------------------------------------------------------

    normalized = df[
        [
            "text",
            "language",
            "intent",
            "sentiment",
            "priority",
            "category",
            "source",
        ]
    ].copy()

    return normalized


# ============================================================
# QUALITY CLEANING
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    print(
        f"\nRows before cleaning: {len(df)}"
    )

    # Remove empty text
    df = df[
        df["text"].str.len() > 2
    ]

    # Remove exact duplicates
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "text",
            "intent",
        ]
    )

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed}"
    )

    # Remove invalid intents
    df = df[
        df["intent"].notna()
    ]

    # Reset index
    df = df.reset_index(drop=True)

    print(
        f"Rows after cleaning: "
        f"{len(df)}"
    )

    return df


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_statistics(
    df: pd.DataFrame,
):

    print("\n")
    print("=" * 70)
    print("BITEXT DATASET STATISTICS")
    print("=" * 70)

    print(
        f"\nTotal samples: {len(df)}"
    )

    print(
        f"Unique texts: "
        f"{df['text'].nunique()}"
    )

    print(
        f"Categories: "
        f"{df['category'].nunique()}"
    )

    print(
        f"Intents: "
        f"{df['intent'].nunique()}"
    )

    print("\n" + "-" * 70)
    print("CATEGORY DISTRIBUTION")
    print("-" * 70)

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("INTENT DISTRIBUTION")
    print("-" * 70)

    print(
        df["intent"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("SAMPLE DATA")
    print("-" * 70)

    print(
        df[
            [
                "text",
                "category",
                "intent",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BITEXT DATASET ADAPTER")
    print("=" * 70)

    # Load
    df = load_bitext_dataset()

    # Validate
    validate_dataset(df)

    # Normalize
    df = normalize_bitext(df)

    # Clean
    df = clean_dataset(df)

    # Statistics
    print_statistics(df)

    # Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nNormalized dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()