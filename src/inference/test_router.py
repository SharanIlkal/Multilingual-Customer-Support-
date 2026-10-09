from src.models.predict import load_model
from src.inference.intent_router import IntentRouter

def main():
    print("=" * 70)
    print("INTENT ROUTER TEST")
    print("=" * 70)

    (
        model,
        vocab,
        label_to_id,
        id_to_label,
    ) = load_model()

    router = IntentRouter(
        model,
        vocab,
        id_to_label,
    )

    test_messages = [
        "I want to track my order",
        "How can I recover my password?",
        "I want a refund for my purchase",
        "Can I get a refund?",
        "I want to check your refund policy",
        "asdfgh random message",
    ]

    for message in test_messages:
        result = router.classify(message)
        print()
        print("-" * 70)
        print(f"Message    : {message}")
        print(f"Action     : {result['action']}")
        print(f"Intent     : {result['intent']}")
        print(f"Category   : {result['category']}")
        print(f"Confidence : " f"{result['confidence'] * 100:.2f}%")

if __name__ == "__main__":
    main()