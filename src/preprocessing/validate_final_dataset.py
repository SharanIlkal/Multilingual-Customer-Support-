from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed/final")
SPLITS = ["train", "validation", "test"]

def main():
    dfs = {
        split: pd.read_csv(DATA_DIR / f"{split}.csv")
        for split in SPLITS
    }
    print("=" * 70)
    print("FINAL DATASET INTEGRITY CHECK")
    print("=" * 70)
    print("\nSIZES")
    for split, df in dfs.items():
        print(f"{split:10}: {len(df):,}")
    print("\nLANGUAGE DISTRIBUTION")
    for split, df in dfs.items():
        print(f"\n{split.upper()}")
        print(df["language"].value_counts().sort_index().to_string())
    print("\nMISSING VALUES")
    for split, df in dfs.items():
        print(f"\n{split.upper()}")
        print(df.isna().sum().to_string())
    print("\nEMPTY TEXT")
    for split, df in dfs.items():
        empty = (df["text"].astype(str).str.strip().eq("").sum())
        print(f"{split:10}: {empty}")
    print("\nDUPLICATE TEXTS WITHIN SPLIT")
    for split, df in dfs.items():
        duplicates = df["text"].duplicated().sum()
        print(f"{split:10}: {duplicates:,}")
    print("\nLEAKAGE")
    train_texts = set(dfs["train"]["text"])
    validation_texts = set(dfs["validation"]["text"])
    test_texts = set(dfs["test"]["text"])
    print("train-validation:", len(train_texts & validation_texts))
    print("train-test:", len(train_texts & test_texts))
    print("validation-test:", len(validation_texts & test_texts))
    print("\nINTENT COUNTS")
    for split, df in dfs.items():
        print(f"\n{split.upper()} | " f"Unique intents: {df['intent'].nunique()}")
        print(df["intent"].value_counts().sort_index().to_string())
    print("\nCATEGORY COUNTS")
    for split, df in dfs.items():
        print(f"\n{split.upper()} | " f"Unique categories: {df['category'].nunique()}")
        print(df["category"].value_counts().sort_index().to_string())
    print("\nCOLUMNS")
    for split, df in dfs.items():
        print(f"{split:10}: " f"{list(df.columns)}")
    print("\n" + "=" * 70)
    print("INTEGRITY CHECK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()