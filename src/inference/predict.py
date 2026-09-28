import json
from pathlib import Path

import torch

from src.models.bilstm import BiLSTMClassifier
from src.training.bilstm_tokenizer import SimpleTokenizer


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "bilstm"

MODEL_PATH = MODEL_DIR / "best_model.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
LABEL_PATH = MODEL_DIR / "label_encoder.json"


class BiLSTMPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------
        # Load checkpoint
        # --------------------------------------------------

        self.checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.max_length = self.checkpoint["max_length"]

        # --------------------------------------------------
        # Load tokenizer
        # --------------------------------------------------

        self.tokenizer = SimpleTokenizer.load(
            VOCAB_PATH
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
        # Load model
        # --------------------------------------------------

        self.model = BiLSTMClassifier(
            vocab_size=self.checkpoint["vocab_size"],
            embedding_dim=self.checkpoint["embedding_dim"],
            hidden_dim=self.checkpoint["hidden_dim"],
            num_classes=self.checkpoint["num_classes"],
            padding_idx=0,
            dropout=0.3,
        )

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        assert len(self.tokenizer.word2idx) == (
            self.checkpoint["vocab_size"]
        )

        assert len(self.label2idx) == (
            self.checkpoint["num_classes"]
        )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    def predict(self, text):

        if not text or not text.strip():
            raise ValueError(
                "Input text cannot be empty."
            )

        # EXACT SAME TOKENIZATION USED DURING EVALUATION
        input_ids = self.tokenizer.encode(
            text,
            max_length=self.max_length,
        )

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():

            logits, _ = self.model(
                input_tensor
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
    print("MULTILINGUAL CUSTOMER SUPPORT INFERENCE")
    print("=" * 70)

    predictor = BiLSTMPredictor()

    print(
        f"\nDevice: {predictor.device}"
    )

    print(
        f"Vocabulary: "
        f"{predictor.checkpoint['vocab_size']}"
    )

    print(
        f"Max length: "
        f"{predictor.max_length}"
    )

    print(
        f"Classes: "
        f"{predictor.checkpoint['num_classes']}"
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

            result = predictor.predict(text)

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