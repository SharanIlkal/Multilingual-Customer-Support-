from pathlib import Path
import time
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/multilingual")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANGUAGES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128

def load_model():
    print("=" * 70)
    print("LOADING TRANSLATION MODEL")
    print("=" * 70)
    print(f"Model  : {MODEL_NAME}")
    print(f"Device : {DEVICE}")
    print(f"Batch  : {BATCH_SIZE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model

def translate_batch(
    texts,
    target_language,
    tokenizer,
    model,
):
    tokenizer.src_lang = LANGUAGES["en"]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(DEVICE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(LANGUAGES[target_language])
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=1,
            early_stopping=True,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True,)

def get_checkpoint_path(split, language):
    return CHECKPOINT_DIR / f"{split}_{language}.csv"

def translate_dataframe(
    df,
    target_language,
    split,
    tokenizer,
    model,
):
    checkpoint_path = get_checkpoint_path(split,target_language,)
    if checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        translations = checkpoint["text"].tolist()
        start_index = len(translations)
        print(f"Resuming {split}-{target_language} " f"from {start_index}/{len(df)}")
    else:
        translations = []
        start_index = 0
        print(f"Starting {split}-{target_language} " f"from 0/{len(df)}")
    texts = df["text"].tolist()
    total = len(texts)
    for start in range(
        start_index,
        total,
        BATCH_SIZE,
    ):
        end = min(start + BATCH_SIZE, total,)
        batch = texts[start:end]
        translated = translate_batch(
            batch,
            target_language,
            tokenizer,
            model,
        )
        translations.extend(translated)
        checkpoint = df.iloc[:end].copy()
        checkpoint["text"] = translations
        checkpoint["language"] = target_language
        checkpoint["source"] = (checkpoint["source"].astype(str) + "_translated")
        checkpoint.to_csv(checkpoint_path, index=False,)
        print(f"{target_language}: " f"{end}/{total}",end="\r",)
    print()
    result = df.copy()
    result["text"] = translations
    result["language"] = target_language
    result["source"] = (result["source"].astype(str)+ "_translated")
    return result

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer, model = load_model()
    for split in [
        "train",
        "validation",
        "test",
    ]:
        input_path = DATA_DIR / f"{split}.csv"
        df = pd.read_csv(input_path)
        print("\n" + "=" * 70)
        print(f"PROCESSING: {split}")
        print("=" * 70)
        print(f"Samples: {len(df)}")
        english_output = (OUTPUT_DIR / f"{split}_en.csv" )
        if english_output.exists():
            print(f"English already exists: " f"{english_output}")
        else:
            english = df.copy()
            english["language"] = "en"
            english.to_csv(english_output, index=False,)
            print(f"Saved: {english_output}")

        for language in [
            "hi",
            "kn",
            "mr",
            "ta",
            "te",
        ]:
            output_path = (OUTPUT_DIR/ f"{split}_{language}.csv")
            if output_path.exists():
                print(f"Skipping {language}: " f"final file already exists")
                continue
            start_time = time.time()
            translated = translate_dataframe(
                df,
                language,
                split,
                tokenizer,
                model,
            )
            translated.to_csv(output_path, index=False,)
            elapsed = time.time() - start_time
            print(f"Saved: {output_path}")
            print(f"Time: " f"{elapsed / 60:.2f} minutes")
    print("\n" + "=" * 70)
    print("TRANSLATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
