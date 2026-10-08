from pathlib import Path
import pandas as pd
INPUT_PATH = Path("data/processed/clean_dataset.csv")

def classify_length(text):
    words = len(str(text).split())
    if words <= 5:
        return "short"
    elif words <= 12:
        return "medium"
    else:
        return "long"

def main():
    df = pd.read_csv(INPUT_PATH)
    df["word_count"] = df["text"].str.split().str.len()
    df["phrase_length"] = df["text"].apply(classify_length)
    print("=" * 70)
    print("PHRASE LENGTH ANALYSIS")
    print("=" * 70)
    print("\nOverall:")
    print(df["phrase_length"].value_counts().to_string())
    print("\nPercentage:")
    print((df["phrase_length"].value_counts(normalize=True) * 100).round(2).to_string())
    print("\nWord-count statistics:")
    print(df["word_count"].describe().to_string())
    print("\nPer-intent distribution:")
    print(
        pd.crosstab(
            df["intent"],
            df["phrase_length"]
        ).to_string()
    )
    print("=" * 70)
    
if __name__ == "__main__":
    main()