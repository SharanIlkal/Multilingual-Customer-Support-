from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path("data/processed/multilingual")
OUTPUT_DIR = Path("data/processed/clean_multilingual")
LANGUAGES = ["hi", "kn", "mr", "ta", "te"]
SPLITS = {
    "train": 19699,
    "validation": 2465,
    "test": 2471,
}

def tokenize(text):
    return re.findall(r"\S+", str(text).strip())

def repeated_token_ratio(text):
    tokens = tokenize(text.lower())
    if not tokens:
        return 1.0
    counts = pd.Series(tokens).value_counts()
    repeated = sum(
        count - 1
        for count in counts
        if count > 1
    )
    return repeated / len(tokens)


def longest_repeated_sequence(text):
    tokens = tokenize(text.lower())
    if len(tokens) < 4:
        return 0
    max_repeat = 1
    for size in range(2, min(8, len(tokens) // 2 + 1)):
        for i in range(len(tokens) - size):
            phrase = tokens[i:i + size]
            for j in range(i + size, len(tokens) - size + 1):
                if tokens[j:j + size] == phrase:
                    max_repeat = max(max_repeat, size)
    return max_repeat


def english_ratio(text):
    text = str(text)
    letters = [
        c for c in text
        if c.isalpha()
    ]
    if not letters:
        return 0.0
    english = sum(
        "a" <= c.lower() <= "z"
        for c in letters
    )
    return english / len(letters)

def script_ratio(text, language):
    patterns = {
        "hi": r"[\u0900-\u097F]",
        "mr": r"[\u0900-\u097F]",
        "kn": r"[\u0C80-\u0CFF]",
        "ta": r"[\u0B80-\u0BFF]",
        "te": r"[\u0C00-\u0C7F]",
    }

    chars = [
        c for c in str(text)
        if c.isalpha()
    ]

    if not chars:
        return 0.0

    pattern = patterns[language]
    matched = sum(
        bool(re.match(pattern, c))
        for c in chars
    )
    return matched / len(chars)

def extract_placeholders(text):
    text = str(text)
    placeholders = []
    placeholders.extend(re.findall(r"\{\{[^{}]+\}\}", text))
    placeholders.extend(re.findall(r"\{[^{}]+\}", text))
    placeholders.extend(re.findall(r"#[0-9]+", text))
    return placeholders


def quality_flags(english, translated,language):
    flags = []
    en_tokens = tokenize(english)
    tr_tokens = tokenize(translated)
    if not tr_tokens:
        flags.append("EMPTY")
        return flags
    if len(en_tokens) >= 8 and len(tr_tokens) <= 2:
        flags.append("TOO_SHORT")
    if len(en_tokens) <= 6 and len(tr_tokens) > 30:
        flags.append("TOO_LONG")
    repeat_ratio = repeated_token_ratio(translated)
    if repeat_ratio >= 0.50:
        flags.append("HIGH_REPETITION")
    if longest_repeated_sequence(translated) >= 5:
        flags.append("REPEATED_SEQUENCE")
    if script_ratio(translated, language) < 0.30:
        flags.append("LOW_TARGET_SCRIPT")
    if english_ratio(translated) > 0.70:
        flags.append("HIGH_ENGLISH")

    digit_map = str.maketrans(
        "०१२३४५६७८९"
        "೦೧೨೩೪೫೬೭೮೯"
        "௦௧௨௩௪௫௬௭௮௯"
        "౦౧౨౩౪౫౬౭౮౯",
        "0123456789"
        "0123456789"
        "0123456789"
        "0123456789",
    )

    english_normalized = str(english)
    translated_normalized = str(translated).translate(digit_map)
    en_numbers = re.findall(r"\d+", english_normalized)
    tr_numbers = re.findall(r"\d+", translated_normalized)
    if en_numbers:
        if not all(
            number in tr_numbers
            for number in en_numbers
        ):
            flags.append("MISSING_NUMBER")
    return flags

def process_file(split, language):
    input_path = BASE_DIR / f"{split}_{language}.csv"
    df = pd.read_csv(input_path)
    flags = []
    for _, row in df.iterrows():
        row_flags = quality_flags(
            row["text"],
            row["text"],
            language
        )
        flags.append(row_flags)
    english_path = BASE_DIR / f"{split}_en.csv"
    english_df = pd.read_csv(english_path)
    final_flags = []
    for i in range(len(df)):
        row_flags = quality_flags(
            english_df.iloc[i]["text"],
            df.iloc[i]["text"],
            language
        )

        final_flags.append("|".join(row_flags) if row_flags else "")
    df["quality_flags"] = final_flags
    bad_mask = df["quality_flags"] != ""
    clean_df = df.loc[~bad_mask].copy()
    flagged_df = df.loc[bad_mask].copy()
    return clean_df, flagged_df

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    audit_dir = OUTPUT_DIR / "quality_audit"
    audit_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    total_rows = 0
    total_bad = 0
    print("=" * 70)
    print("MULTILINGUAL TRANSLATION QUALITY FILTER")
    print("=" * 70)

    for split, expected_rows in SPLITS.items():
        for language in LANGUAGES:
            clean_df, flagged_df = process_file(split, language)
            total = len(clean_df) + len(flagged_df)
            total_rows += total
            total_bad += len(flagged_df)
            clean_path = (OUTPUT_DIR / f"{split}_{language}.csv")
            clean_df.to_csv(
                clean_path,
                index=False,
                encoding="utf-8-sig"
            )
            flagged_path = (audit_dir / f"{split}_{language}_flagged.csv")
            flagged_df.to_csv(
                flagged_path,
                index=False,
                encoding="utf-8-sig"
            )

            print(
                f"{split:10} {language.upper()} | "
                f"Total: {total:5} | "
                f"Clean: {len(clean_df):5} | "
                f"Flagged: {len(flagged_df):5}"
            )
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total rows checked : {total_rows}")
    print(f"Flagged rows       : {total_bad}")

    if total_rows:
        print(f"Flag rate          : " f"{total_bad / total_rows * 100:.2f}%")
    print(f"\nClean datasets:")
    print(OUTPUT_DIR)
    print(f"\nFlagged datasets:")
    print(audit_dir)

if __name__ == "__main__":
    main()