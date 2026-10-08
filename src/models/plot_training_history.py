import os
import json
import matplotlib.pyplot as plt

HISTORY_PATH = "results/training_history.json"
OUTPUT_DIR = "results/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)
epochs = history["epoch"]
train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_acc = history["train_accuracy"]
val_acc = history["val_accuracy"]

plt.figure(figsize=(10, 6))
plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss",
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BiLSTM + Attention Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_path = os.path.join(OUTPUT_DIR, "training_validation_loss.png",)
plt.savefig(loss_path, dpi=300)
plt.close()
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_acc,
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    epochs,
    val_acc,
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("BiLSTM + Attention Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
accuracy_path = os.path.join(OUTPUT_DIR, "training_validation_accuracy.png",)
plt.savefig(accuracy_path, dpi=300)
plt.close()
print("=" * 70)
print("TRAINING HISTORY PLOTS COMPLETE")
print("=" * 70)
print("\nLoss plot:")
print(loss_path)
print("\nAccuracy plot:")
print(accuracy_path)