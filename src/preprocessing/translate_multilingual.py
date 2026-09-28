import os

os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SOURCE_LANGUAGE = "eng_Latn"

TARGET_LANGUAGES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
}

PILOT_PER_INTENT = 20
BATCH_SIZE = 4
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 256

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "data" / "final"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "intermediate"
    / "multilingual_translations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PLACEHOLDER HANDLING
# ============================================================

PLACEHOLDER_PATTERN = re.compile(r"\{\{.*?\}\}")


def protect_placeholders(text: str):
    """
    Remove placeholders before translation.

    Example:

        I need help with {{Order Number}} status

    becomes:

        I need help with  status

    The original placeholders are returned separately
    and restored after translation.
    """

    placeholders = PLACEHOLDER_PATTERN.findall(text)

    protected_text = PLACEHOLDER_PATTERN.sub(
        "",
        text,
    )

    # Clean up extra whitespace created by removing placeholders.
    protected_text = re.sub(
        r"\s+",
        " ",
        protected_text,
    ).strip()

    return protected_text, placeholders


def restore_placeholders(text: str, placeholders):
    """
    Restore placeholders into the translated text.

    For one or more placeholders, append them to the
    translated sentence rather than asking NLLB to preserve
    special placeholder tokens.
    """

    if not placeholders:
        return text

    restored_text = text.strip()

    for placeholder in placeholders:
        restored_text = (
            restored_text.rstrip()
            + " "
            + placeholder
        )

    return restored_text


def validate_placeholders(original_text: str, translated_text: str):
    """
    Ensure that every {{...}} placeholder from the original
    exists exactly in the translated text.
    """

    original_placeholders = PLACEHOLDER_PATTERN.findall(
        original_text
    )

    translated_placeholders = PLACEHOLDER_PATTERN.findall(
        translated_text
    )

    return original_placeholders == translated_placeholders


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING NLLB-200 FROM LOCAL CACHE")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print("Offline cache mode: ENABLED")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()
    model.generation_config.max_length = None

    print(f"Device: {device}")
    print("NLLB model loaded successfully from cache.")
    print()

    return tokenizer, model, device


# ============================================================
# TRANSLATION
# ============================================================

def translate_batch(
    texts,
    tokenizer,
    model,
    device,
    target_language,
):
    """
    Translate a batch of English texts into target language.

    Placeholders are protected before translation and restored
    afterward.
    """

    protected_texts = []
    placeholder_sets = []

    for text in texts:

        protected, placeholders = protect_placeholders(
            str(text)
        )

        protected_texts.append(protected)
        placeholder_sets.append(placeholders)

    tokenizer.src_lang = SOURCE_LANGUAGE

    encoded = tokenizer(
        protected_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        target_language
    )

    with torch.no_grad():

        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
        )

    translations = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    restored_translations = []

    for translated_text, placeholders in zip(
        translations,
        placeholder_sets,
    ):

        restored = restore_placeholders(
            translated_text,
            placeholders,
        )

        restored_translations.append(restored)

    return restored_translations


# ============================================================
# BUILD PILOT DATASET
# ============================================================

def select_pilot_data(df):
    """
    Select the first PILOT_PER_INTENT examples
    from every intent.

    Uses groupby().head() so that the original
    columns, including 'intent', are preserved.
    """

    selected = (
        df.groupby(
            "intent",
            sort=False,
        )
        .head(PILOT_PER_INTENT)
        .reset_index(drop=True)
    )

    return selected


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    tokenizer,
    model,
    device,
):

    input_file = INPUT_DIR / f"{split_name}.csv"

    output_file = (
        OUTPUT_DIR
        / f"{split_name}_translations.csv"
    )

    print("=" * 70)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 70)

    df = pd.read_csv(input_file)

    print(f"Original rows: {len(df):,}")

    required_columns = [
        "text",
        "language",
        "intent",
        "category",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    pilot = select_pilot_data(df)

    print(
        f"Pilot rows per language: "
        f"{len(pilot):,}"
    )

    print(
        f"Intents: "
        f"{pilot['intent'].nunique()}"
    )

    all_rows = []

    # --------------------------------------------------------
    # Process every target language
    # --------------------------------------------------------

    for language_code, target_language in TARGET_LANGUAGES.items():

        print()
        print(
            f"Translating → "
            f"{language_code}"
        )

        texts = pilot["text"].astype(str).tolist()

        # ----------------------------------------------------
        # English
        # ----------------------------------------------------

        if language_code == "en":

            translations = texts

        # ----------------------------------------------------
        # Other languages
        # ----------------------------------------------------

        else:

            translations = []

            for start in range(
                0,
                len(texts),
                BATCH_SIZE,
            ):

                batch = texts[
                    start:start + BATCH_SIZE
                ]

                batch_translations = translate_batch(
                    batch,
                    tokenizer,
                    model,
                    device,
                    target_language,
                )

                translations.extend(
                    batch_translations
                )

                completed = min(
                    start + BATCH_SIZE,
                    len(texts),
                )

                print(
                    f"  Progress: "
                    f"{completed}/{len(texts)}",
                    end="\r",
                )

        # ----------------------------------------------------
        # Build rows
        # ----------------------------------------------------

        for index, (_, row) in enumerate(
            pilot.iterrows()
        ):

            original_text = str(row["text"])
            translated_text = str(
                translations[index]
            )

            placeholders_ok = (
                validate_placeholders(
                    original_text,
                    translated_text,
                )
            )

            # Fail immediately if a placeholder was lost.
            if not placeholders_ok:

                raise ValueError(
                    "\nPlaceholder mismatch detected!\n"
                    f"Language: {language_code}\n"
                    f"Original: {original_text}\n"
                    f"Translated: {translated_text}\n"
                )

            all_rows.append(
                {
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "source_language": "en",
                    "target_language": language_code,
                    "translation_source": "NLLB-200",
                    "intent": row["intent"],
                    "category": row["category"],
                    "source": row["source"],
                }
            )

        print(
            f"  Completed {language_code}: "
            f"{len(translations):,} rows"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = pd.DataFrame(all_rows)

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Saved: {output_file}"
    )

    print(
        f"Total rows: {len(result):,}"
    )

    print(
        "\nLanguage distribution:"
    )

    print(
        result["target_language"]
        .value_counts()
        .sort_index()
    )

    print()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MULTILINGUAL TRANSLATION PILOT")
    print("=" * 70)
    print()

    print(
        f"Languages: "
        f"{', '.join(TARGET_LANGUAGES.keys())}"
    )

    print(
        f"Examples per intent: "
        f"{PILOT_PER_INTENT}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Model: "
        f"{MODEL_NAME}"
    )

    print(
        "HF download: DISABLED"
    )

    print()

    tokenizer, model, device = load_model()

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        process_split(
            split_name,
            tokenizer,
            model,
            device,
        )

    print()
    print("=" * 70)
    print("MULTILINGUAL TRANSLATION PILOT COMPLETE")
    print("=" * 70)
    print()

    print(
        "Expected rows per split:"
    )

    print(
        "27 intents × 20 examples × 6 languages = 3,240"
    )


if __name__ == "__main__":
    main()