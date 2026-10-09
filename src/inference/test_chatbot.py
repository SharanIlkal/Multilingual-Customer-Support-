from src.models.predict import load_model
from src.inference.chatbot import SupportChatbot

def main():

    print("=" * 70)
    print("END-TO-END SUPPORT CHATBOT TEST")
    print("=" * 70)
    model, vocab, label_to_id, id_to_label = load_model()
    chatbot = SupportChatbot(model, vocab, id_to_label)
    messages = ["I want to track my refund", "R1001",]

    for message in messages:
        print()
        print("-" * 70)
        print(f"USER: {message}")
        result = chatbot.process_message(message)
        print(f"BOT : {result['response']}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: " f"{result['confidence'] * 100:.2f}%")
        print(f"Action: {result['action']}")
        if "workflow_status" in result:
            print(f"Workflow: " f"{result['workflow_status']}")

if __name__ == "__main__":
    main()