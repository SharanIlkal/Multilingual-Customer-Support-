import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from bilstm_attention import (
    BiLSTMAttentionClassifier,
    TextDataset,
    MAX_LENGTH,
    BATCH_SIZE,
    MODEL_PATH,
    DEVICE,
    PAD_TOKEN,
)

TEST_PATH = "data/processed/final/test.csv"

@torch.no_grad()
def get_predictions(model, dataloader):
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
    print("BiLSTM + ATTENTION PER-LANGUAGE EVALUATION")
    print("=" * 70)
    test_df = pd.read_csv(TEST_PATH)
    print(f"\nTest samples: {len(test_df):,}")
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    vocab = checkpoint["vocab"]
    label_to_id = checkpoint["label_to_id"]
    id_to_label = checkpoint["id_to_label"]
    config = checkpoint["config"]
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

    print("\nGenerating predictions...")
    predictions = get_predictions(model, test_loader,)
    test_df["predicted_id"] = predictions
    test_df["predicted_intent"] = [
        id_to_label[int(prediction)]
        for prediction in predictions
    ]
    results = []
    print("\n" + "=" * 70)
    print("PER-LANGUAGE RESULTS")
    print("=" * 70)

    for language in sorted(test_df["language"].unique()):
        language_df = test_df[test_df["language"] == language]
        y_true = [
            label_to_id[label]
            for label in language_df["intent"]
        ]

        y_pred = language_df["predicted_id"].tolist()
        accuracy = accuracy_score(y_true,y_pred,)
        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        results.append(
            {
                "language": language,
                "samples": len(language_df),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

        print(
            f"{language.upper():<8}"
            f"Samples: {len(language_df):>5,} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Weighted F1: {weighted_f1:.4f}"
        )
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/language_evaluation.csv", index=False,)

    print("\nResults saved to:" "\nresults/language_evaluation.csv")
    print("\n" + "=" * 70)
    print("LANGUAGE EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()