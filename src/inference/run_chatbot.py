from src.models.predict import load_model
from src.inference.chatbot import SupportChatbot


def main():

    print("=" * 70)
    print("MULTILINGUAL CUSTOMER SUPPORT CHATBOT")
    print("=" * 70)

    print("\nLoading model...")

    model, vocab, label_to_id, id_to_label = load_model()

    chatbot = SupportChatbot(
        model,
        vocab,
        id_to_label
    )

    print("\nChatbot is ready.")
    print("Type 'exit' to quit.")
    print("Type 'reset' to start a new conversation.")

    while True:

        print()
        message = input("YOU: ").strip()

        if not message:
            continue

        if message.lower() in {"exit", "quit"}:
            print("\nBOT: Thank you for contacting customer support.")
            break

        if message.lower() == "reset":
            chatbot.reset()
            print("\nBOT: Conversation has been reset. How can I help you?")
            continue

        try:
            result = chatbot.process_message(message)

            print(f"\nBOT: {result['response']}")

            # Development information
            print(
                f"[Intent: {result['intent']} | "
                f"Confidence: {result['confidence'] * 100:.2f}% | "
                f"Action: {result['action']}]"
            )

        except Exception as error:
            print(f"\nERROR: {error}")


if __name__ == "__main__":
    main()