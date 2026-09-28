from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "models"

CSV_PATH = OUTPUT_DIR / "model_comparison.csv"
PNG_PATH = OUTPUT_DIR / "model_comparison.png"


def main():

    print("=" * 70)
    print("MULTILINGUAL MODEL COMPARISON")
    print("=" * 70)

    data = {
        "Model": [
            "TF-IDF + Logistic Regression",
            "BiLSTM + Attention",
            "XLM-R Embedding Classifier",
        ],
        "Accuracy": [
            88.71,
            92.34,
            93.73,
        ],
        "Macro F1": [
            88.54,
            92.26,
            93.60,
        ],
        "Weighted F1": [
            88.68,
            92.33,
            93.73,
        ],
        "Test Samples": [
            5118,
            5118,
            5118,
        ],
    }

    df = pd.DataFrame(data)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        CSV_PATH,
        index=False,
    )

    print("\nFinal multilingual results:")
    print(df.to_string(index=False))

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    metrics = [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
    ]

    ax = df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Model")
    ax.set_title(
        "Multilingual Customer Support Model Comparison"
    )

    ax.set_ylim(80, 100)

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nComparison CSV saved to:\n{CSV_PATH}"
    )

    print(
        f"\nComparison chart saved to:\n{PNG_PATH}"
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()