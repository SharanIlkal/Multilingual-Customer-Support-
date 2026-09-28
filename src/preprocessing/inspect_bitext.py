from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "data/raw/external_datasets/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("=" * 70)
    print("BITEXT DATASET INSPECTION")
    print("=" * 70)

    df = pd.read_csv(
        DATASET_PATH,
        encoding="utf-8"
    )

    print(f"\nRows: {len(df)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nUnique values per column:")

    for column in df.columns:
        print(
            f"{column}: "
            f"{df[column].nunique()}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()