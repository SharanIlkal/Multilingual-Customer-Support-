from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT
    / "data"
    / "final"
    / "multilingual"
)


def analyze_split(split_name: str):

    path = DATASET_DIR / f"{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} DATASET STATISTICS")
    print("=" * 70)

    print(f"File           : {path}")
    print(f"Total samples  : {len(df):,}")
    print(f"Unique texts   : {df['text'].nunique():,}")
    print(f"Unique intents : {df['intent'].nunique():,}")

    print("\nLanguage distribution:")
    print(
        df["language"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nIntent distribution:")
    print(
        df["intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nText length statistics:")

    lengths = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print(
        lengths.describe()
        .to_string()
    )

    print("\nMissing values:")

    print(
        df.isna()
        .sum()
        .to_string()
    )


def main():

    print("=" * 70)
    print("MULTILINGUAL CUSTOMER SUPPORT DATASET STATISTICS")
    print("=" * 70)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        analyze_split(split)

    print("\n" + "=" * 70)
    print("STATISTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()