from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# DIRECTORIES
# ============================================================

ENGLISH_DIR = ROOT / "data" / "final"

QUALITY_DIR = (
    ROOT
    / "data"
    / "intermediate"
    / "multilingual_quality"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "final"
    / "multilingual"
)


LANGUAGES = [
    "en",
    "hi",
    "kn",
    "ta",
    "te",
    "mr",
]


# ============================================================
# EXPECTED SCHEMA
# ============================================================

TRAINING_COLUMNS = [
    "text",
    "language",
    "intent",
    "category",
    "source",
]


QUALITY_COLUMNS = [
    "original_text",
    "translated_text",
    "source_language",
    "target_language",
    "translation_source",
    "intent",
    "category",
    "source",
    "quality_issues",
    "quality_status",
]


# ============================================================
# LOAD ENGLISH BENCHMARK
# ============================================================

def load_english_split(filename: str) -> pd.DataFrame:

    path = ENGLISH_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"English dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = [
        "text",
        "intent",
        "category",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}"
        )

    result = pd.DataFrame({
        "text": (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        ),

        "language": "en",

        "intent": (
            df["intent"]
            .fillna("")
            .astype(str)
            .str.strip()
        ),

        "category": (
            df["category"]
            .fillna("")
            .astype(str)
            .str.strip()
        ),

        "source": (
            df["source"]
            if "source" in df.columns
            else "bitext"
        ),
    })

    result["source"] = (
        result["source"]
        .fillna("bitext")
        .astype(str)
        .str.strip()
    )

    result = result[
        (result["text"] != "")
        & (result["intent"] != "")
    ]

    result = result.drop_duplicates(
        subset=["text", "intent"]
    )

    return result.reset_index(drop=True)


# ============================================================
# LOAD QUALITY FILE
# ============================================================

def load_quality_split(
    split_name: str
) -> pd.DataFrame:

    path = (
        QUALITY_DIR
        / f"{split_name}_quality.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Quality file not found:\n{path}"
        )

    df = pd.read_csv(path)

    missing = [
        column
        for column in QUALITY_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}"
        )

    return df


# ============================================================
# PREPARE ACCEPTED TRANSLATIONS
# ============================================================

def prepare_translations(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # Only quality-passed translations
    df = df[
        df["quality_status"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        == "pass"
    ].copy()

    # Remove empty translations
    df["translated_text"] = (
        df["translated_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["target_language"] = (
        df["target_language"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["intent"] = (
        df["intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["category"] = (
        df["category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["source"] = (
        df["source"]
        .fillna("bitext")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["translated_text"] != "")
        & (df["target_language"].isin(LANGUAGES))
        & (df["intent"] != "")
    ].copy()

    # Convert to training schema
    result = pd.DataFrame({
        "text": df["translated_text"],
        "language": df["target_language"],
        "intent": df["intent"],
        "category": df["category"],
        "source": df["source"],
    })

    # Remove duplicate translated samples
    result = result.drop_duplicates(
        subset=[
            "text",
            "language",
            "intent",
        ]
    )

    return result.reset_index(drop=True)


# ============================================================
# BUILD MULTILINGUAL DATASET
# ============================================================

def build_split(
    split_name: str,
    filename: str
):

    print("\n" + "=" * 60)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 60)

    # --------------------------------------------------------
    # English benchmark
    # --------------------------------------------------------

    english_df = load_english_split(filename)

    print(
        f"English benchmark : {len(english_df):,}"
    )

    # --------------------------------------------------------
    # Translation quality data
    # --------------------------------------------------------

    quality_df = load_quality_split(
        split_name
    )

    print(
        f"Translation rows  : {len(quality_df):,}"
    )

    # --------------------------------------------------------
    # Accepted translations
    # --------------------------------------------------------

    translations_df = prepare_translations(
        quality_df
    )

    print(
        f"Passed translations: {len(translations_df):,}"
    )

    # --------------------------------------------------------
    # Keep English benchmark only once
    #
    # Quality files also contain English rows.
    # We use the full original English benchmark and
    # only non-English translated rows from quality files.
    # --------------------------------------------------------

    non_english = translations_df[
        translations_df["language"] != "en"
    ].copy()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    multilingual_df = pd.concat(
        [
            english_df,
            non_english,
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Final duplicate protection
    # --------------------------------------------------------

    before = len(multilingual_df)

    multilingual_df = multilingual_df.drop_duplicates(
        subset=[
            "text",
            "language",
            "intent",
        ]
    ).reset_index(drop=True)

    removed = before - len(multilingual_df)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        multilingual_df,
        split_name
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    multilingual_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Save provenance
    # --------------------------------------------------------

    provenance = multilingual_df.copy()

    provenance_path = (
        OUTPUT_DIR
        / f"{split_name}_provenance.csv"
    )

    provenance.to_csv(
        provenance_path,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        f"Duplicate rows removed: {removed:,}"
    )

    print(
        f"Final records         : {len(multilingual_df):,}"
    )

    print("\nLanguage distribution:")

    language_counts = (
        multilingual_df["language"]
        .value_counts()
        .sort_index()
    )

    for language, count in language_counts.items():

        print(
            f"  {language}: {count:,}"
        )

    print(
        f"\nTraining file: {output_path}"
    )

    print(
        f"Provenance file: {provenance_path}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    split_name: str
):

    missing = [
        column
        for column in TRAINING_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{split_name}: Missing columns: {missing}"
        )

    if df.empty:

        raise ValueError(
            f"{split_name}: Dataset is empty."
        )

    if df["text"].isna().any():

        raise ValueError(
            f"{split_name}: Null text values found."
        )

    if df["intent"].isna().any():

        raise ValueError(
            f"{split_name}: Null intent values found."
        )

    invalid_languages = (
        set(df["language"].unique())
        - set(LANGUAGES)
    )

    if invalid_languages:

        raise ValueError(
            f"{split_name}: Invalid languages: "
            f"{sorted(invalid_languages)}"
        )

    empty_text = (
        df["text"]
        .astype(str)
        .str.strip()
        == ""
    )

    if empty_text.any():

        raise ValueError(
            f"{split_name}: Empty text values found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "MULTILINGUAL CUSTOMER SUPPORT DATASET BUILDER"
    )
    print("=" * 60)

    print(
        "\nUsing quality-filtered NLLB translations."
    )

    print(
        "English benchmark is kept intact."
    )

    print(
        "Only quality-passed non-English translations "
        "are added."
    )

    splits = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }

    for split_name, filename in splits.items():

        build_split(
            split_name,
            filename
        )

    print("\n" + "=" * 60)
    print(
        "MULTILINGUAL DATASET BUILD COMPLETE"
    )
    print("=" * 60)

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()