from pathlib import Path
import pandas as pd

BASE_DIR = Path("data/processed/multilingual")
LANGUAGES = ["hi", "kn", "mr", "ta", "te"]
TARGET_INTENTS = [
    "check_invoice",
    "get_invoice",
    "track_order",
    "track_refund",
    "payment_issue",
    "recover_password",
    "change_order",
    "cancel_order",
    "delivery_period",
    "check_cancellation_fee",
]

LENGTH_GROUPS = ["short", "medium", "long"]
SAMPLES_PER_GROUP = 5

def classify_length(text):
    words = str(text).split()
    if len(words) <= 5:
        return "short"
    elif len(words) <= 12:
        return "medium"
    else:
        return "long"

def main():
    output_dir = Path("data/processed/audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("CREATING TRANSLATION SEMANTIC AUDIT DATASET")
    print("=" * 70)
    audit_rows = []

    for intent in TARGET_INTENTS:
        print(f"\nProcessing intent: {intent}")
        en_path = BASE_DIR / "train_en.csv"
        en = pd.read_csv(en_path)
        subset = en[en["intent"] == intent].copy()
        if subset.empty:
            print("  No samples found")
            continue
        subset["length_group"] = subset["text"].apply(classify_length)
        for language in LANGUAGES:
            translated_path = BASE_DIR / f"train_{language}.csv"
            translated = pd.read_csv(translated_path)
            for length_group in LENGTH_GROUPS:
                candidates = subset[subset["length_group"] == length_group]
                if candidates.empty:
                    continue
                candidates = candidates.sample(min(SAMPLES_PER_GROUP, len(candidates)), random_state=42)
                for idx in candidates.index:
                    english_row = en.loc[idx]
                    translated_row = translated.loc[idx]
                    audit_rows.append({
                        "intent": intent,
                        "length_group": length_group,
                        "language": language,
                        "english_text": english_row["text"],
                        "translated_text": translated_row["text"],
                    })

    audit_df = pd.DataFrame(audit_rows)
    output_path = output_dir / "translation_semantic_audit.csv"
    audit_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )
    print("\n" + "=" * 70)
    print("AUDIT DATASET CREATED")
    print("=" * 70)
    print(f"Rows: {len(audit_df)}")
    print(f"Intents: {audit_df['intent'].nunique()}")
    print(f"Languages: {audit_df['language'].nunique()}")
    print("\nLength distribution:")
    print(audit_df["length_group"].value_counts().sort_index())
    print("\nLanguage distribution:")
    print(audit_df["language"].value_counts().sort_index())
    print(f"\nSaved to:")
    print(output_path)

if __name__ == "__main__":
    main()