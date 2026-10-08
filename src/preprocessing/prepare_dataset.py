from pathlib import Path
import pandas as pd

RAW_PATH = Path(
    "data/raw/external_datasets/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "clean_dataset.csv"

def main():
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns={"instruction": "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df["language"] = "en"
    df["source"] = "bitext"
    df = df[["text", "language", "intent", "category", "source"]]
    df = df.drop_duplicates(subset=["text", "intent"])
    df = df.reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)
    print(f"Original samples : {len(pd.read_csv(RAW_PATH))}")
    print(f"Clean samples    : {len(df)}")
    print(f"Intents          : {df['intent'].nunique()}")
    print(f"Categories       : {df['category'].nunique()}")
    print(f"Languages        : {df['language'].unique().tolist()}")
    print(f"Output           : {OUTPUT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()