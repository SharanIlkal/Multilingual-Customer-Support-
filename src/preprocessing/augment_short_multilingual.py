import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/final/multilingual/train.csv")
OUTPUT_PATH = Path("data/final/multilingual/train_augmented.csv")

TARGET_PER_INTENT = 30
LANGUAGES = ["hi", "kn", "mr", "ta", "te"]


def main():
    df = pd.read_csv(INPUT_PATH)

    df["word_count"] = df["text"].astype(str).str.split().str.len()

    short_df = df[
        (df["language"].isin(LANGUAGES))
        & (df["word_count"] <= 5)
    ].copy()

    augmented = []

    for language in LANGUAGES:
        for intent in sorted(df["intent"].unique()):

            current = short_df[
                (short_df["language"] == language)
                & (short_df["intent"] == intent)
            ]

            if len(current) == 0:
                continue

            needed = TARGET_PER_INTENT - len(current)

            if needed <= 0:
                continue

            samples = current.sample(
                n=needed,
                replace=True,
                random_state=42,
            ).copy()

            augmented.append(samples)

    if augmented:
        augmented_df = pd.concat(augmented, ignore_index=True)
        augmented_df = augmented_df.drop(columns=["word_count"])

        result = pd.concat(
            [
                df.drop(columns=["word_count"]),
                augmented_df,
            ],
            ignore_index=True,
        )
    else:
        result = df.drop(columns=["word_count"])

    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("SHORT MULTILINGUAL DATA AUGMENTATION")
    print("=" * 70)
    print(f"Original samples : {len(df)}")
    print(f"Augmented samples: {len(result) - len(df)}")
    print(f"Final samples    : {len(result)}")
    print(f"Saved to         : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()