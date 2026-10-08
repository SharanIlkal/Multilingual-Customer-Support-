from pathlib import Path
import pandas as pd
import random
import re

BASE_DIR = Path("data/processed/multilingual")
SPLITS = {
    "train": 19699,
    "validation": 2465,
    "test": 2471,
}
LANGUAGES = ["en", "hi", "kn", "mr", "ta", "te"]
REQUIRED_COLUMNS = ["text", "language", "intent", "category", "source"]

def classify_length(text):
    words = str(text).split()
    if len(words) <= 5:
        return "short"
    elif len(words) <= 12:
        return "medium"
    else:
        return "long"


def script_ratio(text, pattern):
    text = str(text)
    chars = [c for c in text if not c.isspace() and c.isalpha()]
    if not chars:
        return 0.0
    matched = sum(bool(re.match(pattern, c)) for c in chars)
    return matched / len(chars)
SCRIPT_PATTERNS = {
    "en": r"[A-Za-z]",
    "hi": r"[\u0900-\u097F]",
    "mr": r"[\u0900-\u097F]",
    "kn": r"[\u0C80-\u0CFF]",
    "ta": r"[\u0B80-\u0BFF]",
    "te": r"[\u0C00-\u0C7F]",
}

def validate_file(path, split, language, expected_rows):
    print(f"\n{'=' * 70}")
    print(f"{split.upper()} - {language.upper()}")
    print(f"{'=' * 70}")
    df = pd.read_csv(path)
    errors = []
    if len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(df)}")
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    if errors:
        for error in errors:
            print("ERROR:", error)
        return False
    wrong_language = (df["language"] != language).sum()

    if wrong_language:
        errors.append(f"Wrong language labels: {wrong_language}")
    missing_text = df["text"].isna().sum()
    empty_text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    if missing_text:
        errors.append(f"Missing text: {missing_text}")
    if empty_text:
        errors.append(f"Empty text: {empty_text}")
    for column in ["intent", "category", "source"]:
        missing = df[column].isna().sum()
        if missing:
            errors.append(f"Missing {column}: {missing}")
    duplicate_text = df["text"].astype(str).duplicated().sum()
    print(f"Rows: {len(df)}")
    print(f"Duplicate translated texts: {duplicate_text}")
    df["length_group"] = df["text"].apply(classify_length)
    length_counts = df["length_group"].value_counts()
    print("\nPhrase length distribution:")
    for group in ["short", "medium", "long"]:
        print(f"  {group:7}: " f"{length_counts.get(group, 0)}")
    pattern = SCRIPT_PATTERNS[language]
    ratios = df["text"].astype(str).apply(lambda x: script_ratio(x, pattern))
    low_script = (ratios < 0.30).sum()
    print(f"\nLow expected-script ratio (<30%): " f"{low_script}")
    if low_script > len(df) * 0.10:
        errors.append(f"High number of low-script translations: " f"{low_script}")
    print("\nSample translations:")

    sample_df = df.sample(min(6, len(df)),random_state=42)
    for _, row in sample_df.iterrows():
        print(f"\n[{row['length_group'].upper()}]")
        print(f"Intent : {row['intent']}")
        print(f"Text   : {row['text']}")
    if errors:
        print("\nSTATUS: FAILED")
        for error in errors:
            print("ERROR:", error)
        return False
    print("\nSTATUS: PASSED")
    return True

def main():
    print("\n" + "=" * 70)
    print("MULTILINGUAL TRANSLATION QUALITY VALIDATION")
    print("=" * 70)
    all_passed = True
    for split, expected_rows in SPLITS.items():
        for language in LANGUAGES:
            filename = f"{split}_{language}.csv"
            path = BASE_DIR / filename
            if not path.exists():
                print(f"\nERROR: Missing file: {filename}")
                all_passed = False
                continue
            passed = validate_file(
                path,
                split,
                language,
                expected_rows
            )
            if not passed:
                all_passed = False
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    if all_passed:
        print("ALL TRANSLATION FILES PASSED BASIC QUALITY VALIDATION")
    else:
        print("SOME FILES REQUIRE ATTENTION")

if __name__ == "__main__":
    main()