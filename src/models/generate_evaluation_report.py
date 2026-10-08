import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from bilstm_attention import (
    BiLSTMAttentionClassifier,
    TextDataset,
    build_vocab,
    MAX_LENGTH,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
)

TEST_PATH = "data/processed/final/test.csv"
MODEL_PATH = "models/bilstm_attention.pt"
REPORT_DIR = "results/evaluation"
os.makedirs(REPORT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")
print("=" * 70)
print("FINAL BILSTM + ATTENTION EVALUATION REPORT")
print("=" * 70)
df = pd.read_csv(TEST_PATH)
texts = df["text"].astype(str).tolist()
labels = df["intent"].tolist()
classes = sorted(df["intent"].unique())
label_to_id = {label: i for i, label in enumerate(classes)}
id_to_label = {i: label for label, i in label_to_id.items()}
y_true = np.array([label_to_id[label] for label in labels])
print(f"\nTest samples: {len(df)}")
print(f"Number of classes: {len(classes)}")
print("\nBuilding vocabulary...")
train_df = pd.read_csv("data/processed/final/train.csv")
vocab = build_vocab(train_df["text"].astype(str).tolist(),)
print(f"Vocabulary size: {len(vocab)}")
print("\nEncoding test data...")
test_dataset = TextDataset(
    df,
    vocab,
    label_to_id,
    MAX_LENGTH,
)

X_test = torch.stack([
    test_dataset[i][0]
    for i in range(len(test_dataset))
])
print(f"Encoded test shape: {X_test.shape}")
print("\nLoading model...")
model = BiLSTMAttentionClassifier(
    vocab_size=len(vocab),
    embedding_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=len(classes),
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

checkpoint = torch.load(MODEL_PATH,map_location=DEVICE,)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")
print("\nGenerating predictions...")
with torch.no_grad():
    X_test = X_test.to(DEVICE)
    output = model(X_test)
    logits = output[0]
    predictions = torch.argmax(
        logits,
        dim=1
    ).cpu().numpy()
accuracy = accuracy_score(y_true, predictions)

macro_f1 = f1_score(
    y_true,
    predictions,
    average="macro",
)

weighted_f1 = f1_score(
    y_true,
    predictions,
    average="weighted",
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)
print(f"\nAccuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")
report = classification_report(
    y_true,
    predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(REPORT_DIR, "classification_report.csv")
report_df.to_csv(report_path)
print("\nClassification report saved to:")
print(report_path)
cm = confusion_matrix(
    y_true,
    predictions,
    labels=list(range(len(classes))),
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
)

cm_path = os.path.join(REPORT_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_path)
print("\nConfusion matrix saved to:")
print(cm_path)
metrics = {
    "model": "BiLSTM + Attention",
    "test_samples": len(df),
    "num_classes": len(classes),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
}

metrics_path = os.path.join(REPORT_DIR, "overall_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f,indent=4,)
prediction_df = df.copy()
prediction_df["true_intent"] = [id_to_label[i] for i in y_true]
prediction_df["predicted_intent"] = [id_to_label[i] for i in predictions]
prediction_df["correct"] = (prediction_df["true_intent"]== prediction_df["predicted_intent"])
prediction_path = os.path.join(REPORT_DIR, "test_predictions.csv")
prediction_df.to_csv(prediction_path, index=False,)
print("\nOverall metrics saved to:")
print(metrics_path)
print("\nPredictions saved to:")
print(prediction_path)
print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)