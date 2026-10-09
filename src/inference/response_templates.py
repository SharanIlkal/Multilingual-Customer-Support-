RESPONSES = {

    "en": {
        "order_number":
            "Sure. Please provide your order number.",

        "refund_id":
            "Sure. Please provide your refund ID.",

        "invoice_id":
            "Sure. Please provide your invoice number.",

        "unknown_order":
            "I could not find that order.",

        "unknown_refund":
            "I could not find that refund.",
    },

    "hi": {
        "order_number":
            "ज़रूर। कृपया अपना ऑर्डर नंबर बताएं।",

        "refund_id":
            "ज़रूर। कृपया अपना रिफंड आईडी बताएं।",

        "invoice_id":
            "ज़रूर। कृपया अपना इनवॉइस नंबर बताएं।",

        "unknown_order":
            "मुझे वह ऑर्डर नहीं मिला।",

        "unknown_refund":
            "मुझे वह रिफंड नहीं मिला।",
    },

    "kn": {
        "order_number":
            "ಖಂಡಿತ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಆರ್ಡರ್ ಸಂಖ್ಯೆಯನ್ನು ನೀಡಿ.",

        "refund_id":
            "ಖಂಡಿತ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ರಿಫಂಡ್ ಐಡಿಯನ್ನು ನೀಡಿ.",

        "invoice_id":
            "ಖಂಡಿತ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇನ್ವಾಯ್ಸ್ ಸಂಖ್ಯೆಯನ್ನು ನೀಡಿ.",

        "unknown_order":
            "ಆ ಆರ್ಡರ್ ನನಗೆ ಕಂಡುಬಂದಿಲ್ಲ.",

        "unknown_refund":
            "ಆ ರಿಫಂಡ್ ನನಗೆ ಕಂಡುಬಂದಿಲ್ಲ.",
    },

    "mr": {
        "order_number":
            "नक्की. कृपया तुमचा ऑर्डर नंबर सांगा.",

        "refund_id":
            "नक्की. कृपया तुमचा रिफंड आयडी सांगा.",

        "invoice_id":
            "नक्की. कृपया तुमचा इनव्हॉइस नंबर सांगा.",

        "unknown_order":
            "मला तो ऑर्डर सापडला नाही.",

        "unknown_refund":
            "मला तो रिफंड सापडला नाही.",
    },

    "ta": {
        "order_number":
            "நிச்சயமாக. தயவுசெய்து உங்கள் ஆர்டர் எண்ணை வழங்கவும்.",

        "refund_id":
            "நிச்சயமாக. தயவுசெய்து உங்கள் ரீஃபண்ட் ஐடியை வழங்கவும்.",

        "invoice_id":
            "நிச்சயமாக. தயவுசெய்து உங்கள் இன்வாய்ஸ் எண்ணை வழங்கவும்.",

        "unknown_order":
            "அந்த ஆர்டர் கிடைக்கவில்லை.",

        "unknown_refund":
            "அந்த ரீஃபண்ட் கிடைக்கவில்லை.",
    },

    "te": {
        "order_number":
            "తప్పకుండా. దయచేసి మీ ఆర్డర్ నంబర్‌ను ఇవ్వండి.",

        "refund_id":
            "తప్పకుండా. దయచేసి మీ రీఫండ్ IDని ఇవ్వండి.",

        "invoice_id":
            "తప్పకుండా. దయచేసి మీ ఇన్వాయిస్ నంబర్‌ను ఇవ్వండి.",

        "unknown_order":
            "ఆ ఆర్డర్ నాకు కనిపించలేదు.",

        "unknown_refund":
            "ఆ రీఫండ్ నాకు కనిపించలేదు.",
    },
}


def get_response(
    language,
    key,
    **kwargs
):

    language_responses = RESPONSES.get(language, RESPONSES["en"])
    template = language_responses.get(key, RESPONSES["en"].get(key, ""))
    return template.format(**kwargs)