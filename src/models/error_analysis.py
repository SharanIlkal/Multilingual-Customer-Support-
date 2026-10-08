import os
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from bilstm_attention import (
    BiLSTMAttentionClassifier,
    TextDataset,
    build_vocab,
    build_label_mapping,
    PAD_TOKEN,
    UNK_TOKEN,
    MAX_LENGTH,
    BATCH_SIZE,
    MODEL_PATH,
    DEVICE,
)

TEST_PATH = "data/processed/final/test.csv"

OUTPUT_DIR = "results/error_analysis"
ERRORS_PATH = os.path.join(OUTPUT_DIR,"bilstm_test_errors.csv",)
CONFUSION_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.csv",)

@torch.no_grad()
def get_predictions(model,dataloader,):
    model.eval()
    predictions = []
    for input_ids, _ in dataloader:
        input_ids = input_ids.to(DEVICE)
        logits, _ = model(input_ids)
        batch_predictions = torch.argmax(logits, dim=1,)
        predictions.extend(batch_predictions.cpu().numpy())
    return predictions

def main():
    print("=" * 70)
    print("BiLSTM + ATTENTION ERROR ANALYSIS")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True,)
    print("\nLoading test dataset...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test samples: {len(test_df):,}")
    print("\nLoading best BiLSTM model...")
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    vocab = checkpoint["vocab"]
    label_to_id = checkpoint["label_to_id"]
    id_to_label = checkpoint["id_to_label"]
    config = checkpoint["config"]
    test_dataset = TextDataset(
        test_df,
        vocab,
        label_to_id,
        MAX_LENGTH,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=len(label_to_id),
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_idx=vocab[PAD_TOKEN],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    print("\nGenerating predictions...")
    predictions = get_predictions(model, test_loader,)
    test_df["predicted_intent"] = [
        id_to_label[int(prediction)]
        for prediction in predictions
    ]
    test_df["correct"] = (test_df["intent"]== test_df["predicted_intent"])
    errors = test_df[~test_df["correct"]].copy()
    print(f"\nCorrect predictions: " f"{test_df['correct'].sum():,}")
    print(f"Incorrect predictions: " f"{len(errors):,}")
    print(f"Error rate: " f"{len(errors) / len(test_df):.4%}")
    error_columns = [
        "sample_id",
        "text",
        "language",
        "intent",
        "predicted_intent",
        "category",
    ]
    errors[error_columns].to_csv(
        ERRORS_PATH,
        index=False,
        encoding="utf-8",
    )
    print(f"\nErrors saved to:" f"\n{ERRORS_PATH}")
    print("\n" + "=" * 70)
    print("ERRORS BY LANGUAGE")
    print("=" * 70)
    language_stats = (test_df
        .groupby("language")
        .agg(
            total=("correct", "size"),
            errors=("correct", lambda x: (~x).sum(),),
        )
    )

    language_stats["error_rate"] = (language_stats["errors"]/ language_stats["total"])
    print(
        language_stats
        .sort_values("error_rate", ascending=False,)
        .to_string()
    )
    print("\n" + "=" * 70)
    print("ERRORS BY INTENT")
    print("=" * 70)
    intent_stats = (
        test_df
        .groupby("intent")
        .agg(
            total=("correct", "size"),
            errors=("correct", lambda x: (~x).sum(),),
        )
    )

    intent_stats["error_rate"] = (intent_stats["errors"]/ intent_stats["total"])
    print(
        intent_stats
        .sort_values("error_rate",ascending=False,)
        .to_string()
    )
    print("\n" + "=" * 70)
    print("TOP CONFUSION PAIRS")
    print("=" * 70)
    confusion_pairs = (
        errors
        .groupby(
            [
                "intent",
                "predicted_intent",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count",ascending=False,)
    )

    print(
        confusion_pairs
        .head(20)
        .to_string(index=False)
    )

    labels = sorted(label_to_id.keys())
    matrix = confusion_matrix(
        test_df["intent"],
        test_df["predicted_intent"],
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    confusion_df.to_csv(CONFUSION_PATH, encoding="utf-8",)
    print(f"\nConfusion matrix saved to:" f"\n{CONFUSION_PATH}")
    print("\n" + "=" * 70)
    print("SAMPLE ERRORS")
    print("=" * 70)
    sample_errors = (
        errors[
            [
                "text",
                "language",
                "intent",
                "predicted_intent",
            ]
        ]
        .head(30)
    )

    for index, row in sample_errors.iterrows():
        print(f"\n[{index}] " f"Language: {row['language']}")
        print(f"Text      : {row['text']}")
        print(f"Actual    : {row['intent']}")
        print(f"Predicted : " f"{row['predicted_intent']}")
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS COMPLETE")
        print("=" * 70)

if __name__ == "__main__":
    main()