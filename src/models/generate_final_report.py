import os
import json
import pandas as pd

FINAL_DIR = "results/final_report"
EVAL_DIR = "results/evaluation"
ERROR_DIR = "results/error_analysis"
os.makedirs(FINAL_DIR, exist_ok=True)

dataset_stats = {
    "total_samples": 147461,
    "train_samples": 117926,
    "validation_samples": 14754,
    "test_samples": 14781,
    "languages": ["English", "Hindi", "Kannada", "Marathi", "Tamil", "Telugu"],
    "num_languages": 6,
    "num_intents": 27,
    "num_categories": 11,
}

model_results = [
    {
        "model": "TF-IDF + Logistic Regression",
        "test_accuracy": 0.9224,
        "macro_f1": 0.9205,
        "weighted_f1": 0.9224,
    },
    {
        "model": "XLM-R Embeddings + Logistic Regression",
        "test_accuracy": 0.9593,
        "macro_f1": 0.9570,
        "weighted_f1": 0.9592,
    },
    {
        "model": "BiLSTM + Attention",
        "test_accuracy": 0.9801,
        "macro_f1": 0.9793,
        "weighted_f1": 0.9802,
    },
]

language_results = [
    {
        "language": "English",
        "samples": 2471,
        "accuracy": 0.9899,
        "macro_f1": 0.9888,
        "weighted_f1": 0.9899,
    },
    {
        "language": "Hindi",
        "samples": 2465,
        "accuracy": 0.9797,
        "macro_f1": 0.9789,
        "weighted_f1": 0.9798,
    },
    {
        "language": "Kannada",
        "samples": 2460,
        "accuracy": 0.9780,
        "macro_f1": 0.9773,
        "weighted_f1": 0.9782,
    },
    {
        "language": "Marathi",
        "samples": 2457,
        "accuracy": 0.9723,
        "macro_f1": 0.9718,
        "weighted_f1": 0.9724,
    },
    {
        "language": "Tamil",
        "samples": 2459,
        "accuracy": 0.9776,
        "macro_f1": 0.9762,
        "weighted_f1": 0.9776,
    },
    {
        "language": "Telugu",
        "samples": 2469,
        "accuracy": 0.9830,
        "macro_f1": 0.9824,
        "weighted_f1": 0.9830,
    },
]


translation_quality = {
    "total_translated_samples_checked": 123175,
    "flagged_samples": 349,
    "flag_rate_percent": 0.28,
    "clean_samples": 122826,
    "languages": [
        "Hindi",
        "Kannada",
        "Marathi",
        "Tamil",
        "Telugu",
    ],
    "intent_mismatches": 0,
    "category_mismatches": 0,
}

error_analysis = {
    "test_samples": 14781,
    "correct_predictions": 14487,
    "incorrect_predictions": 294,
    "error_rate_percent": 1.989,
    "best_language": "English",
    "best_language_accuracy": 0.9899,
    "lowest_language_accuracy": 0.9723,
    "lowest_language": "Marathi",
    "highest_error_intent": "cancel_order",
    "highest_error_intent_error_rate": 0.056856,
}

training = {
    "model": "BiLSTM + Attention",
    "epochs": 15,
    "embedding_dimension": 128,
    "hidden_dimension": 128,
    "lstm_layers": 2,
    "dropout": 0.3,
    "max_sequence_length": 32,
    "batch_size": 256,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "best_epoch": 15,
    "best_validation_accuracy": 0.9814,
    "best_validation_macro_f1": 0.9808,
}

key_findings = [
    "The final dataset contains 147,461 multilingual samples across 6 languages.",
    "The dataset covers 27 customer-support intents grouped into 11 categories.",
    "The BiLSTM with Attention achieved the best overall performance.",
    "BiLSTM + Attention achieved 98.01% test accuracy and 97.93% Macro F1.",
    "The BiLSTM outperformed the TF-IDF baseline by 5.77 percentage points in accuracy.",
    "XLM-R embeddings achieved 95.93% test accuracy.",
    "English achieved the highest language-specific accuracy at 98.99%.",
    "Marathi achieved the lowest language-specific accuracy at 97.23%.",
    "Translation validation found zero intent mismatches and zero category mismatches.",
    "Translation quality filtering flagged 349 of 123,175 translated samples.",
    "Most remaining classification errors occurred between semantically similar customer-support intents.",
]

conclusion = (
    "The multilingual customer-support intent classification system "
    "demonstrates strong classification performance across English, Hindi, "
    "Kannada, Marathi, Tamil, and Telugu. Among the evaluated approaches, "
    "the BiLSTM with Attention produced the strongest results, achieving "
    "98.01% test accuracy, 97.93% Macro F1, and 98.02% Weighted F1. "
    "The evaluation also shows strong performance across all six languages, "
    "with language-specific accuracy ranging from 97.23% to 98.99%. "
    "The remaining errors are concentrated primarily in semantically related "
    "intent classes such as invoice retrieval/checking, order modification/"
    "tracking, cancellation/order changes, and refund-related intents."
)

report = {
    "project": "Multilingual Customer Support Intent Classification",
    "dataset": dataset_stats,
    "translation_quality": translation_quality,
    "model_comparison": model_results,
    "language_evaluation": language_results,
    "training_configuration": training,
    "error_analysis": error_analysis,
    "key_findings": key_findings,
    "conclusion": conclusion,
}

json_path = os.path.join(FINAL_DIR, "final_project_report.json")
with open(json_path, "w", encoding="utf-8",) as f:
    json.dump(
        report,
        f,
        indent=4,
        ensure_ascii=False,
    )

model_df = pd.DataFrame(model_results)
model_csv = os.path.join(FINAL_DIR, "model_comparison.csv")
model_df.to_csv(model_csv, index=False,)
language_df = pd.DataFrame(language_results)
language_csv = os.path.join(FINAL_DIR,"language_performance.csv")
language_df.to_csv(language_csv,index=False,)
txt_path = os.path.join(FINAL_DIR,"final_project_report.txt")
with open(txt_path,"w", encoding="utf-8",) as f:
    f.write("=" * 80 + "\n")
    f.write("MULTILINGUAL CUSTOMER SUPPORT " "INTENT CLASSIFICATION\n")
    f.write()
    f.write("=" * 80 + "\n\n")
    f.write("1. DATASET\n")
    f.write("-" * 80 + "\n")
    for key, value in dataset_stats.items():
        f.write(f"{key}: {value}\n")
    f.write("\n2. TRANSLATION QUALITY\n")
    f.write("-" * 80 + "\n")
    for key, value in translation_quality.items():
        f.write(f"{key}: {value}\n")
    f.write("\n3. MODEL COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(model_df.to_string(index=False))
    f.write("\n\n4. LANGUAGE PERFORMANCE\n")
    f.write("-" * 80 + "\n")
    f.write(language_df.to_string(index=False))
    f.write("\n\n5. TRAINING CONFIGURATION\n")
    f.write("-" * 80 + "\n")
    for key, value in training.items():
        f.write(f"{key}: {value}\n")
    f.write("\n6. ERROR ANALYSIS\n")
    f.write("-" * 80 + "\n")

    for key, value in error_analysis.items():
        f.write(f"{key}: {value}\n")
    f.write("\n7. KEY FINDINGS\n")
    f.write("-" * 80 + "\n")

    for i, finding in enumerate(
        key_findings,
        start=1,
    ):
        f.write(f"{i}. {finding}\n")
    f.write("\n8. CONCLUSION\n")
    f.write("-" * 80 + "\n")
    f.write(conclusion)
    f.write("\n")
print("=" * 80)
print("FINAL PROJECT REPORT CREATED")
print("=" * 80)
print(f"\nJSON:" f"\n{json_path}")
print(f"\nModel comparison:" f"\n{model_csv}")
print(f"\nLanguage performance:" f"\n{language_csv}")
print(f"\nHuman-readable report:" f"\n{txt_path}")
print("\nBest Model: BiLSTM + Attention")
print("Test Accuracy: 98.01%")
print("Macro F1: 97.93%")
print("Weighted F1: 98.02%")