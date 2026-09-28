from pathlib import Path
import argparse

import pandas as pd


# ============================================================
# STANDARD DATASET SCHEMA
# ============================================================

STANDARD_COLUMNS = [
    "text",
    "language",
    "intent",
    "sentiment",
    "priority",
    "category",
    "source",
]


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {
    # Text
    "message": "text",
    "query": "text",
    "question": "text",
    "utterance": "text",
    "sentence": "text",
    "text_content": "text",
    "customer_message": "text",
    "customer_query": "text",

    # Language
    "lang": "language",
    "language_code": "language",

    # Intent
    "label": "intent",
    "intent_label": "intent",
    "category_label": "intent",
    "intent_name": "intent",

    # Sentiment
    "emotion": "sentiment",
    "sentiment_label": "sentiment",
    "sentiment_class": "sentiment",

    # Priority
    "urgency": "priority",
    "urgency_level": "priority",
    "priority_level": "priority",

    # Category
    "topic": "category",
    "department": "category",
    "domain": "category",
}


# ============================================================
# LANGUAGE NORMALIZATION
# ============================================================

LANGUAGE_MAP = {
    "english": "en",
    "en": "en",

    "hindi": "hi",
    "hi": "hi",

    "kannada": "kn",
    "kn": "kn",

    "tamil": "ta",
    "ta": "ta",

    "telugu": "te",
    "te": "te",

    "marathi": "mr",
    "mr": "mr",

    "malayalam": "ml",
    "ml": "ml",

    "bengali": "bn",
    "bangla": "bn",
    "bn": "bn",

    "gujarati": "gu",
    "gu": "gu",

    "punjabi": "pa",
    "pa": "pa",

    "urdu": "ur",
    "ur": "ur",
}


# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe column names.

    Example:

        Customer Message -> customer_message
        Intent Label     -> intent_label
        Language         -> language
    """

    df = df.copy()

    normalized_columns = []

    for column in df.columns:

        column = str(column).strip().lower()

        column = (
            column
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        normalized_columns.append(column)

    df.columns = normalized_columns

    # Apply aliases
    df = df.rename(columns=COLUMN_ALIASES)

    return df


# ============================================================
# DETECT TEXT COLUMN
# ============================================================

def detect_text_column(df: pd.DataFrame) -> str | None:
    """
    Automatically detect the most likely text column.
    """

    candidates = [
        "text",
        "message",
        "query",
        "question",
        "utterance",
        "sentence",
        "customer_message",
        "customer_query",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# NORMALIZE LANGUAGE
# ============================================================

def normalize_language(value) -> str:

    if pd.isna(value):
        return "unknown"

    value = str(value).strip().lower()

    return LANGUAGE_MAP.get(value, value)


# ============================================================
# ENSURE STANDARD COLUMNS
# ============================================================

def ensure_standard_columns(
    df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if "text" not in df.columns:

        text_column = detect_text_column(df)

        if text_column is None:

            raise ValueError(
                "Could not find a text column.\n"
                "Expected one of:\n"
                "text, message, query, question, "
                "utterance, sentence, customer_message"
            )

        df["text"] = df[text_column]

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    if "language" not in df.columns:

        # Most external datasets may be English-only.
        df["language"] = "en"

    df["language"] = df["language"].apply(
        normalize_language
    )

    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------

    if "intent" not in df.columns:

        df["intent"] = "unknown"

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    if "sentiment" not in df.columns:

        df["sentiment"] = "unknown"

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if "priority" not in df.columns:

        df["priority"] = "unknown"

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if "category" not in df.columns:

        df["category"] = "unknown"

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    df["source"] = source_name

    return df


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text) -> str:

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove leading/trailing spaces
    text = text.strip()

    # Normalize repeated whitespace
    text = " ".join(text.split())

    return text


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    df["text"] = df["text"].apply(clean_text)

    # Remove empty messages
    df = df[df["text"].str.len() > 2]

    # --------------------------------------------------------
    # Clean intent
    # --------------------------------------------------------

    df["intent"] = (
        df["intent"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Clean sentiment
    # --------------------------------------------------------

    df["sentiment"] = (
        df["sentiment"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Clean priority
    # --------------------------------------------------------

    df["priority"] = (
        df["priority"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Clean category
    # --------------------------------------------------------

    df["category"] = (
        df["category"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "text",
            "language",
            "intent",
        ]
    )

    removed = before - len(df)

    print(f"Duplicates removed: {removed}")

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    return df


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

def select_standard_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    for column in STANDARD_COLUMNS:

        if column not in df.columns:

            df[column] = "unknown"

    return df[STANDARD_COLUMNS]


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(df: pd.DataFrame):

    print("\n")
    print("=" * 70)
    print("NORMALIZED DATASET STATISTICS")
    print("=" * 70)

    print(f"\nTotal samples       : {len(df)}")
    print(
        f"Unique texts        : "
        f"{df['text'].nunique()}"
    )

    print(
        f"Languages           : "
        f"{df['language'].nunique()}"
    )

    print(
        f"Intents             : "
        f"{df['intent'].nunique()}"
    )

    print("\n" + "-" * 70)
    print("LANGUAGE DISTRIBUTION")
    print("-" * 70)

    print(
        df["language"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("TOP INTENTS")
    print("-" * 70)

    print(
        df["intent"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print("\n" + "-" * 70)
    print("SENTIMENT DISTRIBUTION")
    print("-" * 70)

    print(
        df["sentiment"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("PRIORITY DISTRIBUTION")
    print("-" * 70)

    print(
        df["priority"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("TEXT LENGTH")
    print("-" * 70)

    text_lengths = (
        df["text"]
        .str.split()
        .str.len()
    )

    print(
        text_lengths.describe()
        .to_string()
    )

    print("\n" + "=" * 70)


# ============================================================
# MAIN NORMALIZATION PIPELINE
# ============================================================

def normalize_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:

    input_path = Path(input_path)
    output_path = Path(output_path)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{input_path}\n\n"
            f"Please place the dataset in "
            f"data/raw/external_datasets/"
        )

    # --------------------------------------------------------
    # Source name
    # --------------------------------------------------------

    source_name = input_path.stem

    print("\n")
    print("=" * 70)
    print("DATASET NORMALIZATION PIPELINE")
    print("=" * 70)

    print(f"\nInput : {input_path}")
    print(f"Output: {output_path}")
    print(f"Source: {source_name}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        input_path,
        encoding="utf-8",
    )

    print(
        f"Original rows: {len(df)}"
    )

    print(
        f"Original columns: "
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df = normalize_column_names(df)

    print(
        f"\nNormalized columns: "
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # Ensure standard schema
    # --------------------------------------------------------

    df = ensure_standard_columns(
        df,
        source_name,
    )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    df = clean_dataset(df)

    # --------------------------------------------------------
    # Select standard columns
    # --------------------------------------------------------

    df = select_standard_columns(df)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_statistics(df)

    print(
        f"\nNormalized dataset saved to:\n"
        f"{output_path}"
    )

    return df


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Normalize customer-support datasets "
            "into the project's standard schema."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to input CSV dataset."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path where normalized dataset "
            "will be saved."
        ),
    )

    args = parser.parse_args()

    normalize_dataset(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()