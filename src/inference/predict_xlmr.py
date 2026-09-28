import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.models.xlmr_embedding_classifier import (
    XLMREmbeddingClassifier,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "xlmr"

CLASSIFIER_PATH = (
    MODEL_DIR / "embedding_classifier.pt"
)

LABEL_PATH = (
    MODEL_DIR / "embedding_label_encoder.json"
)

XLMR_MODEL_NAME = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 64


class XLMRPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------
        # Load tokenizer
        # --------------------------------------------------

        print("\nLoading XLM-R tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            XLMR_MODEL_NAME
        )

        # --------------------------------------------------
        # Load XLM-R
        # --------------------------------------------------

        print("Loading XLM-R model...")

        self.xlmr = AutoModel.from_pretrained(
            XLMR_MODEL_NAME
        )

        self.xlmr.to(self.device)
        self.xlmr.eval()

        # --------------------------------------------------
        # Load classifier checkpoint
        # --------------------------------------------------

        checkpoint = torch.load(
            CLASSIFIER_PATH,
            map_location=self.device,
            weights_only=False,
        )

        # --------------------------------------------------
        # Load labels
        # --------------------------------------------------

        with open(
            LABEL_PATH,
            encoding="utf-8",
        ) as file:

            mapping = json.load(file)

        self.label2idx = mapping["label2idx"]

        self.idx2label = {
            int(key): value
            for key, value in mapping["idx2label"].items()
        }

        # --------------------------------------------------
        # Load classifier
        # --------------------------------------------------

        self.classifier = XLMREmbeddingClassifier(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"],
            dropout=0.3,
        )

        self.classifier.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.classifier.to(self.device)
        self.classifier.eval()

    # ------------------------------------------------------
    # Mean pooling
    # ------------------------------------------------------

    def mean_pooling(
        self,
        last_hidden_state,
        attention_mask,
    ):

        mask = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()

        summed = torch.sum(
            last_hidden_state * mask,
            dim=1,
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )

        return summed / counts

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    def predict(self, text):

        if not text or not text.strip():

            raise ValueError(
                "Input text cannot be empty."
            )

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoded[
            "input_ids"
        ].to(self.device)

        attention_mask = encoded[
            "attention_mask"
        ].to(self.device)

        # --------------------------------------------------
        # XLM-R embedding
        # --------------------------------------------------

        with torch.no_grad():

            outputs = self.xlmr(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            embedding = self.mean_pooling(
                outputs.last_hidden_state,
                attention_mask,
            )

            # --------------------------------------------------
            # Classifier
            # --------------------------------------------------

            logits = self.classifier(
                embedding
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        predicted_id = prediction.item()

        predicted_intent = self.idx2label[
            predicted_id
        ]

        return {
            "text": text,
            "intent": predicted_intent,
            "confidence": confidence.item(),
        }


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 70)
    print("MULTILINGUAL XLM-R INFERENCE")
    print("=" * 70)

    predictor = XLMRPredictor()

    print(
        f"\nDevice: {predictor.device}"
    )

    print(
        f"XLM-R model: {XLMR_MODEL_NAME}"
    )

    print(
        f"Classes: {len(predictor.label2idx)}"
    )

    print("\nEnter customer-support text.")
    print("Type 'exit' to stop.\n")

    while True:

        text = input("Text: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            continue

        try:

            result = predictor.predict(
                text
            )

            print(
                f"\nPredicted Intent : "
                f"{result['intent']}"
            )

            print(
                f"Confidence       : "
                f"{result['confidence']:.4f}"
            )

        except Exception as error:

            print(
                f"\nPrediction error: {error}"
            )

        print()


if __name__ == "__main__":
    main()