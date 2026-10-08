import os
import json
import pandas as pd

OUTPUT_DIR = "results/final_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = [
    {
        "Model": "TF-IDF + Logistic Regression",
        "Test Accuracy": 0.9224,
        "Macro F1": 0.9205,
        "Weighted F1": 0.9224,
    },
    {
        "Model": "XLM-R Embeddings + Logistic Regression",
        "Test Accuracy": 0.9593,
        "Macro F1": 0.9570,
        "Weighted F1": 0.9592,
    },
    {
        "Model": "BiLSTM + Attention",
        "Test Accuracy": 0.9801,
        "Macro F1": 0.9793,
        "Weighted F1": 0.9802,
    },
]


df = pd.DataFrame(results)
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)
print()
print(
    df.to_string(
        index=False,
        formatters={
            "Test Accuracy": "{:.4f}".format,
            "Macro F1": "{:.4f}".format,
            "Weighted F1": "{:.4f}".format,
        },
    )
)

best_index = df["Macro F1"].idxmax()
best_model = df.loc[best_index,"Model"]
best_accuracy = df.loc[best_index, "Test Accuracy"]
best_macro_f1 = df.loc[best_index, "Macro F1"]
best_weighted_f1 = df.loc[best_index,"Weighted F1"]
print()
print("=" * 80)
print("BEST MODEL")
print("=" * 80)
print(f"\nModel           : {best_model}")
print(f"Test Accuracy   : {best_accuracy:.4f}")
print(f"Macro F1        : {best_macro_f1:.4f}")
print(f"Weighted F1     : {best_weighted_f1:.4f}")
baseline_accuracy = df.loc[
    df["Model"] == "TF-IDF + Logistic Regression",
    "Test Accuracy"
].iloc[0]

baseline_macro_f1 = df.loc[
    df["Model"] == "TF-IDF + Logistic Regression",
    "Macro F1"
].iloc[0]

accuracy_improvement = (best_accuracy - baseline_accuracy)
macro_f1_improvement = (best_macro_f1 - baseline_macro_f1)
print()
print("=" * 80)
print("IMPROVEMENT OVER BASELINE")
print("=" * 80)
print(
    f"\nAccuracy improvement : "
    f"{accuracy_improvement:.4f}"
    f" ({accuracy_improvement * 100:.2f} percentage points)"
)

print(
    f"Macro F1 improvement : "
    f"{macro_f1_improvement:.4f}"
    f" ({macro_f1_improvement * 100:.2f} percentage points)"
)

csv_path = os.path.join(OUTPUT_DIR, "final_model_comparison.csv")
df.to_csv(
    csv_path,
    index=False
)

comparison_json = {
    "models": results,
    "best_model": {
        "name": best_model,
        "test_accuracy": float(best_accuracy),
        "macro_f1": float(best_macro_f1),
        "weighted_f1": float(best_weighted_f1),
    },
    "baseline_improvement": {
        "accuracy_points": float(accuracy_improvement * 100),
        "macro_f1_points": float(macro_f1_improvement * 100),
    },
}


json_path = os.path.join(OUTPUT_DIR, "final_model_comparison.json")
with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(comparison_json,f,indent=4)
print()
print("=" * 80)
print("FINAL MODEL COMPARISON COMPLETE")
print("=" * 80)
print(f"\nCSV saved to:" f"\n{csv_path}")
print(f"\nJSON saved to:" f"\n{json_path}")