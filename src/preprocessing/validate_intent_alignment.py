from pathlib import Path
import pandas as pd

BASE_DIR = Path("data/processed/multilingual")
SPLITS = {
    "train": 19699,
    "validation": 2465,
    "test": 2471,
}
LANGUAGES = ["hi", "kn", "mr", "ta", "te"]

def validate_alignment(split, language):
    english_path = BASE_DIR / f"{split}_en.csv"
    translated_path = BASE_DIR / f"{split}_{language}.csv"
    en = pd.read_csv(english_path)
    translated = pd.read_csv(translated_path)
    print(f"\n{'=' * 70}")
    print(f"{split.upper()} - {language.upper()}")
    print(f"{'=' * 70}")
    errors = []

    if len(en) != len(translated):
        errors.append(
            f"Row count mismatch: EN={len(en)}, "
            f"{language.upper()}={len(translated)}"
        )

    if errors:
        for error in errors:
            print("ERROR:", error)
        return False

    intent_match = (en["intent"].astype(str).values == translated["intent"].astype(str).values)
    intent_mismatches = (~intent_match).sum()
    category_match = (en["category"].astype(str).values == translated["category"].astype(str).values)
    category_mismatches = (~category_match).sum()
    source_match = (en["source"].astype(str).values == translated["source"].astype(str).values)
    source_mismatches = (~source_match).sum()
    print(f"Rows: {len(translated)}")
    print(f"Intent mismatches: {intent_mismatches}")
    print(f"Category mismatches: {category_mismatches}")
    print(f"Source mismatches: {source_mismatches}")
    en_counts = en["intent"].value_counts().sort_index()
    translated_counts = translated["intent"].value_counts().sort_index()
    distribution_match = en_counts.equals(translated_counts)
    print(f"Intent distribution preserved: " f"{distribution_match}")
    if not distribution_match:
        errors.append("Intent distribution changed")
    en_categories = en["category"].value_counts().sort_index()
    translated_categories = (translated["category"].value_counts().sort_index())
    category_distribution_match = (en_categories.equals(translated_categories))
    print(f"Category distribution preserved: " f"{category_distribution_match}")
    if not category_distribution_match:
        errors.append("Category distribution changed")
    if intent_mismatches > 0:
        print("\nIntent mismatch examples:")
        mismatch_indices = [
            i for i, match in enumerate(intent_match)
            if not match
        ][:10]
        for i in mismatch_indices:
            print(
                f"\nRow {i}"
                f"\nEN intent       : {en.iloc[i]['intent']}"
                f"\nTranslated intent: {translated.iloc[i]['intent']}"
            )

    if intent_mismatches == 0 and category_mismatches == 0:
        print("\nSTATUS: PASSED")
        return True
    print("\nSTATUS: FAILED")
    if intent_mismatches:
        print(f"  - {intent_mismatches} intent mismatches")
    if category_mismatches:
        print(f"  - {category_mismatches} category mismatches")
    return False

def main():
    print("\n" + "=" * 70)
    print("MULTILINGUAL INTENT ALIGNMENT VALIDATION")
    print("=" * 70)
    all_passed = True
    for split in SPLITS:
        for language in LANGUAGES:
            passed = validate_alignment(split, language)
            if not passed:
                all_passed = False
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    if all_passed:
        print("ALL TRANSLATIONS ARE INTENT-ALIGNED")
    else:
        print("INTENT ALIGNMENT ISSUES FOUND")

if __name__ == "__main__":
    main()