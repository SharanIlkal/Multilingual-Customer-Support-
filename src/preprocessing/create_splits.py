from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/processed/clean_dataset.csv")
OUTPUT_DIR = Path("data/processed")

def classify_length(text):
    words = len(str(text).split())
    if words <= 5:
        return "short"
    elif words <= 12:
        return "medium"
    else:
        return "long"

def stratified_split(group, test_size):
    counts = group["phrase_length"].value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        return train_test_split(
            group,
            test_size=test_size,
            random_state=42,
            stratify=group["phrase_length"],
        )

    return train_test_split(group, test_size=test_size, random_state=42,)

def main():
    df = pd.read_csv(INPUT_PATH)
    df["phrase_length"] = df["text"].apply(classify_length)
    train_parts = []
    val_parts = []
    test_parts = []
    for intent, group in df.groupby("intent"):
        train, temp = stratified_split(group, test_size=0.20,)
        val, test = stratified_split(temp, test_size=0.50,)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = (pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True))
    val_df = (pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True))
    test_df = (pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False,)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False, )
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False,)
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)
    print(f"Total      : {len(df)}")
    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    for name, data in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:
        print(f"\n{name} phrase lengths:")
        print(data["phrase_length"].value_counts().to_string())
        print(f"\n{name} intents:")
        print(data["intent"].value_counts().sort_index().to_string())
    print("=" * 70)

if __name__ == "__main__":
    main()