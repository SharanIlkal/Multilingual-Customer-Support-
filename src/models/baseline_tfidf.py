from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,)
DATA_DIR = Path("data/processed/final")

def main():
    print("=" * 70)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 70)
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train = train["text"].astype(str)
    y_train = train["intent"]
    X_validation = validation["text"].astype(str)
    y_validation = validation["intent"]
    X_test = test["text"].astype(str)
    y_test = test["intent"]
    print("\nDataset sizes:")
    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(validation):,}")
    print(f"Test       : {len(test):,}")
    print("\nBuilding TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=200000,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_validation_tfidf = vectorizer.transform(X_validation)
    X_test_tfidf = vectorizer.transform(X_test)
    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )
    print(
        f"Train matrix: "
        f"{X_train_tfidf.shape}"
    )
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=4.0,
        solver="lbfgs",
    )
    model.fit(X_train_tfidf, y_train)
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    validation_pred = model.predict(X_validation_tfidf)
    validation_accuracy = accuracy_score(y_validation, validation_pred)
    validation_macro_f1 = f1_score(
        y_validation,
        validation_pred,
        average="macro"
    )
    validation_weighted_f1 = f1_score(
        y_validation,
        validation_pred,
        average="weighted"
    )
    print(f"Accuracy    : {validation_accuracy:.4f}")
    print(f"Macro F1    : {validation_macro_f1:.4f}")
    print(f"Weighted F1 : {validation_weighted_f1:.4f}")
    print("\n" + "=" * 70)
    print("TEST")
    print("=" * 70)
    test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test,test_pred)
    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
    )
    test_weighted_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted"
    )
    print(f"Accuracy    : {test_accuracy:.4f}")
    print(f"Macro F1    : {test_macro_f1:.4f}")
    print(f"Weighted F1 : {test_weighted_f1:.4f}")
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(
        classification_report(
            y_test,
            test_pred,
            digits=4,
            zero_division=0,
        )
    )

if __name__ == "__main__":
    main()