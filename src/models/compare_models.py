import json
import os

RESULTS_PATH = "results/model_comparison.json"

def main():
    os.makedirs("results", exist_ok=True)
    results = {
        "tfidf_logistic_regression": {
            "validation": {
                "accuracy": 0.9268,
                "macro_f1": 0.9251,
                "weighted_f1": 0.9268,
            },
            "test": {
                "accuracy": 0.9224,
                "macro_f1": 0.9205,
                "weighted_f1": 0.9224,
            },
        },

        "bilstm_attention": {
            "validation": {
                "accuracy": 0.9814,
                "macro_f1": 0.9808,
                "weighted_f1": 0.9815,
            },
            "test": {
                "accuracy": 0.9801,
                "macro_f1": 0.9793,
                "weighted_f1": 0.9802,
            },
        },

        "xlmr_embeddings_logistic_regression": {
            "validation": {
                "accuracy": 0.9622,
                "macro_f1": 0.9607,
                "weighted_f1": 0.9623,
            },
            "test": {
                "accuracy": 0.9593,
                "macro_f1": 0.9570,
                "weighted_f1": 0.9592,
            },
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8",) as f:
        json.dump(results, f, indent=4,)
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(
        f"{'Model':<40}"
        f"{'Test Accuracy':>15}"
        f"{'Macro F1':>12}"
        f"{'Weighted F1':>14}"
    )
    print("-" * 80)
    for model_name, metrics in results.items():
        display_name = (model_name.replace("_", " ").title())
        test = metrics["test"]
        print(
            f"{display_name:<40}"
            f"{test['accuracy']:>15.4f}"
            f"{test['macro_f1']:>12.4f}"
            f"{test['weighted_f1']:>14.4f}"
        )
    print("=" * 80)
    print(f"\nResults saved to: " f"{RESULTS_PATH}")
    
if __name__ == "__main__":
    main()