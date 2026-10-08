from pathlib import Path
import pandas as pd

CLEAN_DIR = Path("data/processed/clean_multilingual")
ORIGINAL_DIR = Path("data/processed/multilingual")
OUTPUT_DIR = Path("data/processed/final")
LANGUAGES = [
    "en",
    "hi",
    "kn",
    "mr",
    "ta",
    "te",
]

SPLITS = [
    "train",
    "validation",
    "test",
]

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("REBUILDING FINAL MULTILINGUAL DATASET")
    print("=" * 70)
    for split in SPLITS:
        frames = []
        print(f"\n{split.upper()}")
        for language in LANGUAGES:
            if language == "en":
                file_path = (ORIGINAL_DIR / f"{split}_en.csv")
            else:
                file_path = (CLEAN_DIR /f"{split}_{language}.csv")
            df = pd.read_csv(file_path)
            if "sample_id" not in df.columns:
                raise ValueError(f"sample_id missing from {file_path}")
            df["language"] = language
            frames.append(df)
            print(
                f"{language.upper():>3} : "
                f"{len(df):5} rows"
            )

        final_df = pd.concat(frames, ignore_index=True)
        if "quality_flags" in final_df.columns:
            final_df = final_df.drop(columns=["quality_flags"])

        columns = [
            "sample_id",
            "text",
            "language",
            "intent",
            "category",
            "source",
        ]

        final_df = final_df[columns]
        final_df = final_df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        output_path = (OUTPUT_DIR / f"{split}.csv")
        final_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"FINAL {split}: " f"{len(final_df):,} rows")
    print("\n" + "=" * 70)
    print("FINAL DATASET REBUILT")
    print("=" * 70)


if __name__ == "__main__":
    main()