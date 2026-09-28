from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT
    / "data"
    / "intermediate"
    / "multilingual_translations"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "intermediate"
    / "multilingual_quality"
)


LANGUAGE_NAMES = {
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
}

# Unicode ranges
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "mr": (0x0900, 0x097F),  # Devanagari
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
}


def extract_placeholders(text):
    """
    Extract placeholders such as:

        {{Order Number}}
        {{Account Type}}
    """

    return sorted(
        re.findall(r"\{\{.*?\}\}", str(text))
    )


def script_ratio(text, language):
    """
    Percentage of alphabetic characters belonging
    to the expected Indic script.
    """

    if language not in SCRIPT_RANGES:
        return 0.0

    start, end = SCRIPT_RANGES[language]

    letters = [
        char
        for char in str(text)
        if char.isalpha()
    ]

    if not letters:
        return 0.0

    expected = sum(
        start <= ord(char) <= end
        for char in letters
    )

    return expected / len(letters)


def english_ratio(text):
    """
    Approximate proportion of alphabetic characters
    that are Latin/English characters.
    """

    letters = [
        char
        for char in str(text)
        if char.isalpha()
    ]

    if not letters:
        return 0.0

    english = sum(
        ("a" <= char.lower() <= "z")
        for char in letters
    )

    return english / len(letters)


def length_ratio(original, translated):

    original_length = max(
        len(str(original).split()),
        1
    )

    translated_length = max(
        len(str(translated).split()),
        1
    )

    return translated_length / original_length


def evaluate_row(row):

    original = str(row["original_text"])
    translated = str(row["translated_text"])
    language = row["target_language"]

    problems = []

    # -------------------------------------------------
    # Empty translation
    # -------------------------------------------------

    if not translated.strip():
        problems.append("empty_translation")

    # -------------------------------------------------
    # Placeholder preservation
    # -------------------------------------------------

    original_placeholders = extract_placeholders(
        original
    )

    translated_placeholders = extract_placeholders(
        translated
    )

    if original_placeholders != translated_placeholders:
        problems.append(
            "placeholder_mismatch"
        )

    # -------------------------------------------------
    # Language/script validation
    # -------------------------------------------------

    if language in SCRIPT_RANGES:

        ratio = script_ratio(
            translated,
            language
        )

        if ratio < 0.30:
            problems.append(
                "low_expected_script_ratio"
            )

    # -------------------------------------------------
    # Excessive English
    # -------------------------------------------------

    if language != "en":

        ratio = english_ratio(
            translated
        )

        if ratio > 0.65:
            problems.append(
                "excessive_english"
            )

    # -------------------------------------------------
    # Length sanity check
    # -------------------------------------------------

    ratio = length_ratio(
        original,
        translated
    )

    if ratio < 0.20:
        problems.append(
            "translation_too_short"
        )

    if ratio > 4.0:
        problems.append(
            "translation_too_long"
        )

    return problems


def validate_split(split):

    path = (
        INPUT_DIR
        / f"{split}_translations.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    results = []

    for _, row in df.iterrows():

        problems = evaluate_row(row)

        results.append(
            problems
        )

    df["quality_issues"] = [
        ";".join(items)
        for items in results
    ]

    df["quality_status"] = df[
        "quality_issues"
    ].apply(
        lambda value:
        "reject"
        if value
        else "pass"
    )

    return df


def print_report(df, split):

    print()
    print("=" * 60)
    print(f"{split.upper()} QUALITY REPORT")
    print("=" * 60)

    total = len(df)

    passed = (
        df["quality_status"]
        == "pass"
    ).sum()

    rejected = (
        df["quality_status"]
        == "reject"
    ).sum()

    print(f"Total    : {total:,}")
    print(f"Passed   : {passed:,}")
    print(f"Rejected : {rejected:,}")

    print(
        f"Pass rate: "
        f"{passed / total:.2%}"
    )

    print("\nIssues:")

    issue_counts = {}

    for value in df["quality_issues"]:

        if not value:
            continue

        for issue in value.split(";"):

            issue_counts[issue] = (
                issue_counts.get(issue, 0) + 1
            )

    for issue, count in sorted(
        issue_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"  {issue}: {count:,}"
        )

    print("\nRejected examples:")

    rejected_df = df[
        df["quality_status"]
        == "reject"
    ]

    if len(rejected_df) > 0:

        columns = [
            "original_text",
            "translated_text",
            "target_language",
            "intent",
            "quality_issues",
        ]

        print(
            rejected_df[
                columns
            ]
            .head(15)
            .to_string(index=False)
        )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        df = validate_split(
            split
        )

        print_report(
            df,
            split
        )

        output_path = (
            OUTPUT_DIR
            / f"{split}_quality.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"\nSaved: {output_path}"
        )

    print()
    print("=" * 60)
    print("MULTILINGUAL QUALITY CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()