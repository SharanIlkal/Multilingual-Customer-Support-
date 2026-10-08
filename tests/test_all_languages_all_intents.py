"""
Comprehensive Multilingual Customer Support Chatbot Test

Coverage:
    27 intents
    6 languages
    162 intent/language tests
    Multi-turn entity tests
    Multi-turn confirmation tests

Run from project root:

    python tests/test_all_languages_all_intents.py

The test uses the same model-loading logic as the project chatbot runner.
It does not modify production code.
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.inference.chatbot import SupportChatbot
from src.models.predict import load_model

# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = [
    "en",
    "hi",
    "kn",
    "mr",
    "ta",
    "te",
]


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
}


# ============================================================
# TEST PHRASES
# ============================================================

TEST_CASES = {

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    "create_account": {
        "en": "I want to create a new account",
        "hi": "मैं नया अकाउंट बनाना चाहता हूँ",
        "kn": "ನಾನು ಹೊಸ ಖಾತೆಯನ್ನು ರಚಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला नवीन खाते तयार करायचे आहे",
        "ta": "நான் புதிய கணக்கை உருவாக்க விரும்புகிறேன்",
        "te": "నేను కొత్త ఖాతాను సృష్టించాలనుకుంటున్నాను",
    },

    "delete_account": {
        "en": "I want to delete my account",
        "hi": "मैं अपना अकाउंट हटाना चाहता हूँ",
        "kn": "ನನ್ನ ಖಾತೆಯನ್ನು ಅಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला माझे खाते हटवायचे आहे",
        "ta": "எனது கணக்கை நீக்க விரும்புகிறேன்",
        "te": "నా ఖాతాను తొలగించాలనుకుంటున్నాను",
    },

    "edit_account": {
        "en": "I want to edit my account details",
        "hi": "मैं अपने अकाउंट की जानकारी बदलना चाहता हूँ",
        "kn": "ನನ್ನ ಖಾತೆಯ ವಿವರಗಳನ್ನು ಬದಲಾಯಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला माझ्या खात्याचे तपशील बदलायचे आहेत",
        "ta": "எனது கணக்கு விவரங்களை மாற்ற விரும்புகிறேன்",
        "te": "నా ఖాతా వివరాలను మార్చాలనుకుంటున్నాను",
    },

    "recover_password": {
        "en": "I forgot my password",
        "hi": "मैं अपना पासवर्ड भूल गया हूँ",
        "kn": "ನಾನು ನನ್ನ ಪಾಸ್‌ವರ್ಡ್ ಮರೆತಿದ್ದೇನೆ",
        "mr": "मी माझा पासवर्ड विसरलो आहे",
        "ta": "எனது கடவுச்சொல்லை மறந்துவிட்டேன்",
        "te": "నేను నా పాస్‌వర్డ్ మర్చిపోయాను",
    },

    "registration_problems": {
        "en": "I have a problem registering",
        "hi": "मुझे पंजीकरण करने में समस्या हो रही है",
        "kn": "ನೋಂದಣಿ ಮಾಡಲು ನನಗೆ ಸಮಸ್ಯೆ ಇದೆ",
        "mr": "मला नोंदणी करताना समस्या येत आहे",
        "ta": "பதிவு செய்வதில் எனக்கு சிக்கல் உள்ளது",
        "te": "నమోదు చేయడంలో నాకు సమస్య ఉంది",
    },

    "switch_account": {
        "en": "I want to switch to another account",
        "hi": "मैं दूसरे अकाउंट पर स्विच करना चाहता हूँ",
        "kn": "ನಾನು ಬೇರೆ ಖಾತೆಗೆ ಬದಲಾಯಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला दुसऱ्या खात्यावर स्विच करायचे आहे",
        "ta": "நான் மற்றொரு கணக்கிற்கு மாற விரும்புகிறேன்",
        "te": "నేను మరొక ఖాతాకు మారాలనుకుంటున్నాను",
    },


    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    "check_cancellation_fee": {
        "en": "Is there a cancellation fee",
        "hi": "क्या रद्दीकरण शुल्क है",
        "kn": "ರದ್ದತಿ ಶುಲ್ಕ ಇದೆಯೇ",
        "mr": "रद्द करण्याचे शुल्क आहे का",
        "ta": "ரத்து கட்டணம் உள்ளதா",
        "te": "రద్దు రుసుము ఉందా",
    },


    # --------------------------------------------------------
    # CUSTOMER SUPPORT
    # --------------------------------------------------------

    "contact_customer_service": {
        "en": "I want to contact customer service",
        "hi": "मैं ग्राहक सेवा से संपर्क करना चाहता हूँ",
        "kn": "ನಾನು ಗ್ರಾಹಕ ಸೇವೆಯನ್ನು ಸಂಪರ್ಕಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला ग्राहक सेवेशी संपर्क साधायचा आहे",
        "ta": "நான் வாடிக்கையாளர் சேவையை தொடர்பு கொள்ள விரும்புகிறேன்",
        "te": "నేను కಸ್ಟಮర్ ಸರ್ವీస్ ಅನ್ನು సంప్రదించాలనుకుంటున్నాను",
    },

    "contact_human_agent": {
        "en": "I want to talk to a human agent",
        "hi": "मैं मानव सहायता प्रतिनिधि से बात करना चाहता हूँ",
        "kn": "ನಾನು ಮಾನವ ಸಹಾಯ ಪ್ರತಿನಿಧಿಯೊಂದಿಗೆ ಮಾತನಾಡಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला मानव सहाय्य प्रतिनिधीशी बोलायचे आहे",
        "ta": "நான் ஒரு மனித ஆதரவு பிரதிநிதியுடன் பேச விரும்புகிறேன்",
        "te": "నేను మానవ సహాయ ప్రతినిధితో మాట్లాడాలనుకుంటున్నాను",
    },


    # --------------------------------------------------------
    # DELIVERY
    # --------------------------------------------------------

    "delivery_options": {
        "en": "What delivery options are available",
        "hi": "डिलीवरी के कौन से विकल्प उपलब्ध हैं",
        "kn": "ಯಾವ ಡೆಲಿವರಿ ಆಯ್ಕೆಗಳು ಲಭ್ಯವಿವೆ",
        "mr": "डिलिव्हरीचे कोणते पर्याय उपलब्ध आहेत",
        "ta": "என்ன டெலிவரி விருப்பங்கள் உள்ளன",
        "te": "ఏ డెలివరీ ఎంపికలు అందుబాటులో ఉన్నాయి",
    },

    "delivery_period": {
        "en": "How long will delivery take",
        "hi": "डिलीवरी में कितना समय लगेगा",
        "kn": "ಡೆಲಿವರಿಗೆ ಎಷ್ಟು ಸಮಯ ಬೇಕಾಗುತ್ತದೆ",
        "mr": "डिलिव्हरीला किती वेळ लागेल",
        "ta": "டெலிவரிக்கு எவ்வளவு நேரம் ஆகும்",
        "te": "డెలివరీకి ఎంత సమయం పడుతుంది",
    },


    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    "complaint": {
        "en": "I want to make a complaint",
        "hi": "मैं शिकायत करना चाहता हूँ",
        "kn": "ನಾನು ದೂರು ನೀಡಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला तक्रार करायची आहे",
        "ta": "நான் புகார் அளிக்க விரும்புகிறேன்",
        "te": "నేను ఫిర్యాదు చేయాలనుకుంటున్నాను",
    },

    "review": {
        "en": "I want to leave a review",
        "hi": "मैं समीक्षा देना चाहता हूँ",
        "kn": "ನಾನು ವಿಮರ್ಶೆ ನೀಡಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला पुनरावलोकन द्यायचे आहे",
        "ta": "நான் மதிப்புரை வழங்க விரும்புகிறேன்",
        "te": "నేను సమీక్ష ఇవ్వాలనుకుంటున్నాను",
    },


    # --------------------------------------------------------
    # INVOICE
    # --------------------------------------------------------

    "check_invoice": {
        "en": "I want to check my invoice",
        "hi": "मैं अपना इनवॉइस देखना चाहता हूँ",
        "kn": "ನನ್ನ ಇನ್ವಾಯ್ಸ್ ಪರಿಶೀಲಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला माझे इनव्हॉइस तपासायचे आहे",
        "ta": "எனது இன்வாய்ஸை சரிபார்க்க விரும்புகிறேன்",
        "te": "నా ఇన్వాయిస్‌ను తనిఖీ చేయాలనుకుంటున్నాను",
    },

    "get_invoice": {
        "en": "I want to get my invoice",
        "hi": "मैं अपना इनवॉइस प्राप्त करना चाहता हूँ",
        "kn": "ನನ್ನ ಇನ್ವಾಯ್ಸ್ ಪಡೆಯಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला माझे इनव्हॉइस मिळवायचे आहे",
        "ta": "எனது இன்வாய்ஸை பெற விரும்புகிறேன்",
        "te": "నా ఇన్వాయిస్ పొందాలనుకుంటున్నాను",
    },


    # --------------------------------------------------------
    # NEWSLETTER
    # --------------------------------------------------------

    "newsletter_subscription": {
        "en": "I want to manage my newsletter subscription",
        "hi": "मैं अपनी न्यूज़लेटर सदस्यता प्रबंधित करना चाहता हूँ",
        "kn": "ನನ್ನ ಸುದ್ದಿಪತ್ರ ಚಂದಾದಾರಿಕೆಯನ್ನು ನಿರ್ವಹಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला माझी वृत्तपत्र सदस्यता व्यवस्थापित करायची आहे",
        "ta": "எனது நியூஸ்லெட்டர் சந்தாவை நிர்வகிக்க விரும்புகிறேன்",
        "te": "నా న్యూస్‌లెటర్ సభ్యత్వాన్ని నిర్వహించాలనుకుంటున్నాను",
    },


    # --------------------------------------------------------
    # ORDER
    # --------------------------------------------------------

    "cancel_order": {
        "en": "I want to cancel my order",
        "hi": "मेरा ऑर्डर रद्द करना है",
        "kn": "ನನ್ನ ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "माझा ऑर्डर रद्द करायचा आहे",
        "ta": "எனது ஆர்டரை ரத்து செய்ய வேண்டும்",
        "te": "నా ఆర్డర్ రద్దు చేయాలి",
    },

    "change_order": {
        "en": "I want to change my order",
        "hi": "मैं अपना ऑर्डर बदलना चाहता हूँ",
        "kn": "ನನ್ನ ಆರ್ಡರ್ ಬದಲಾಯಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला माझा ऑर्डर बदलायचा आहे",
        "ta": "எனது ஆர்டரை மாற்ற விரும்புகிறேன்",
        "te": "నా ఆర్డర్‌ను మార్చాలనుకుంటున్నాను",
    },

    "place_order": {
        "en": "I want to place an order",
        "hi": "मैं ऑर्डर देना चाहता हूँ",
        "kn": "ನಾನು ಆರ್ಡರ್ ಮಾಡಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला ऑर्डर द्यायचा आहे",
        "ta": "நான் ஆர்டர் செய்ய விரும்புகிறேன்",
        "te": "నేను ఆర్డర్ చేయాలనుకుంటున్నాను",
    },

    "track_order": {
        "en": "Track my order",
        "hi": "मेरा ऑर्डर ट्रैक करें",
        "kn": "ನನ್ನ ಆರ್ಡರ್ ಟ್ರ್ಯಾಕ್ ಮಾಡಿ",
        "mr": "माझा ऑर्डर ट्रॅक करा",
        "ta": "எனது ஆர்டரை கண்காணிக்கவும்",
        "te": "నా ఆర్డర్‌ను ట్రాక్ చేయండి",
    },


    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    "check_payment_methods": {
        "en": "What payment methods are available",
        "hi": "कौन से भुगतान तरीके उपलब्ध हैं",
        "kn": "ಯಾವ ಪಾವತಿ ವಿಧಾನಗಳು ಲಭ್ಯವಿವೆ",
        "mr": "कोणत्या पेमेंट पद्धती उपलब्ध आहेत",
        "ta": "என்ன பேமெண்ட் முறைகள் உள்ளன",
        "te": "ఏ చెల్లింపు పద్ధతులు అందుబాటులో ఉన్నాయి",
    },

    "payment_issue": {
        "en": "I have a problem with my payment",
        "hi": "मेरे भुगतान में समस्या है",
        "kn": "ನನ್ನ ಪಾವತಿಯಲ್ಲಿ ಸಮಸ್ಯೆ ಇದೆ",
        "mr": "माझ्या पेमेंटमध्ये समस्या आहे",
        "ta": "எனது பேமெண்ட்டில் சிக்கல் உள்ளது",
        "te": "నా చెల్లింపులో సమస్య ఉంది",
    },


    # --------------------------------------------------------
    # REFUND
    # --------------------------------------------------------

    "check_refund_policy": {
        "en": "What is the refund policy",
        "hi": "रिफंड की नीति क्या है",
        "kn": "ರಿಫಂಡ್ ನೀತಿ ಏನು",
        "mr": "रिफंडचे धोरण काय आहे",
        "ta": "ரீஃபண்ட் கொள்கை என்ன",
        "te": "రీఫండ్ విధానం ఏమిటి",
    },

    "get_refund": {
        "en": "I want a refund for my order",
        "hi": "मुझे अपने ऑर्डर का रिफंड चाहिए",
        "kn": "ನನ್ನ ಆರ್ಡರ್‌ಗೆ ರಿಫಂಡ್ ಬೇಕು",
        "mr": "मला माझ्या ऑर्डरसाठी रिफंड हवा",
        "ta": "எனது ஆர்டருக்கு ரீஃபண்ட் வேண்டும்",
        "te": "నా ఆర్డర్‌కు రీఫండ్ కావాలి",
    },

    "track_refund": {
        "en": "Track my refund",
        "hi": "मेरा रिफंड ट्रैक करें",
        "kn": "ನನ್ನ ರಿಫಂಡ್ ಟ್ರ್ಯಾಕ್ ಮಾಡಿ",
        "mr": "माझा रिफंड ट्रॅक करा",
        "ta": "எனது ரீஃபண்டை கண்காணிக்கவும்",
        "te": "నా రీఫండ్‌ను ట్రాక్ చేయండి",
    },


    # --------------------------------------------------------
    # SHIPPING ADDRESS
    # --------------------------------------------------------

    "change_shipping_address": {
        "en": "I want to change my shipping address",
        "hi": "मैं अपना शिपिंग पता बदलना चाहता हूँ",
        "kn": "ನನ್ನ ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸವನ್ನು ಬದಲಾಯಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला माझा शिपिंग पत्ता बदलायचा आहे",
        "ta": "எனது ஷிப்பிங் முகவரியை மாற்ற விரும்புகிறேன்",
        "te": "నా షిప్పింగ్ చిరునామాను మార్చాలనుకుంటున్నాను",
    },

    "set_up_shipping_address": {
        "en": "I want to add a shipping address",
        "hi": "मैं शिपिंग पता जोड़ना चाहता हूँ",
        "kn": "ನಾನು ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸವನ್ನು ಸೇರಿಸಲು ಬಯಸುತ್ತೇನೆ",
        "mr": "मला शिपिंग पत्ता जोडायचा आहे",
        "ta": "நான் ஷிப்பிங் முகவரியைச் சேர்க்க விரும்புகிறேன்",
        "te": "నేను షిప్పింగ్ చిరునామాను జోడించాలనుకుంటున్నాను",
    },
}


# ============================================================
# MULTI-TURN TEST CONFIGURATION
# ============================================================

ORDER_ENTITY_INTENTS = {
    "track_order",
    "get_refund",
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_invoice",
    "get_invoice",
}


REFUND_ENTITY_INTENTS = {
    "track_refund",
}


CONFIRMATION_INTENTS = {
    "cancel_order",
    "get_refund",
    "delete_account",
    "change_order",
    "change_shipping_address",
}


ENTITY_VALUES = {
    "order_id": "12",
    "refund_id": "1",
}


CONFIRMATION_WORDS = {
    "en": "yes",
    "hi": "हाँ",
    "kn": "ಹೌದು",
    "mr": "हो",
    "ta": "ஆம்",
    "te": "అవును",
}


# ============================================================
# MODEL LOADING
# ============================================================

def create_chatbot():
    """
    Use the exact same model loader as the real chatbot.
    """

    model, vocab, label_to_id, id_to_label = load_model()

    return SupportChatbot(
        model,
        vocab,
        id_to_label,
    )


# ============================================================
# HELPERS
# ============================================================

def new_chatbot(template):
    """
    Create a fresh chatbot state while reusing the loaded model.
    """

    return SupportChatbot(
        template.router.model,
        template.router.vocab,
        template.router.id_to_label,
    )


def result_ok(result, expected_intent):
    return result.get("intent") == expected_intent


def print_result(
    number,
    total,
    language,
    expected,
    result,
):
    actual = result.get("intent")
    confidence = result.get("confidence", 0.0)

    passed = actual == expected

    status = "PASS" if passed else "FAIL"

    print(
        f"[{number:03d}/{total:03d}] "
        f"{status:<4} "
        f"{language.upper():<2} "
        f"{expected:<30} "
        f"actual={str(actual):<30} "
        f"confidence={confidence * 100:6.2f}%"
    )

    if not passed:
        print(
            f"      Message response: "
            f"{result.get('response', '')}"
        )

    return passed


# ============================================================
# CLASSIFICATION TESTS
# ============================================================

def run_classification_tests(bot):
    """
    Test all 27 intents in all 6 languages.

    27 × 6 = 162 tests.
    """

    print()
    print("=" * 90)
    print("1. INTENT CLASSIFICATION TESTS")
    print("=" * 90)

    expected_count = 27

    if len(TEST_CASES) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} intents, "
            f"but TEST_CASES contains {len(TEST_CASES)}."
        )

    total = len(TEST_CASES) * len(LANGUAGES)

    passed = 0
    failed = []

    number = 0

    for intent, messages in TEST_CASES.items():

        for language in LANGUAGES:

            number += 1

            chatbot = new_chatbot(bot)

            message = messages[language]

            result = chatbot.process_message(message)

            ok = print_result(
                number,
                total,
                language,
                intent,
                result,
            )

            if ok:
                passed += 1
            else:
                failed.append(
                    {
                        "language": language,
                        "expected": intent,
                        "actual": result.get("intent"),
                        "message": message,
                        "response": result.get("response"),
                    }
                )

    return passed, failed, total


# ============================================================
# ENTITY TESTS
# ============================================================

def run_entity_tests(bot):
    """
    Test intents that require order/refund IDs.
    """

    print()
    print("=" * 90)
    print("2. ENTITY FOLLOW-UP TESTS")
    print("=" * 90)

    tests = []

    for intent in sorted(ORDER_ENTITY_INTENTS):
        for language in LANGUAGES:
            tests.append(
                (
                    intent,
                    language,
                    "order_id",
                    ENTITY_VALUES["order_id"],
                )
            )

    for intent in sorted(REFUND_ENTITY_INTENTS):
        for language in LANGUAGES:
            tests.append(
                (
                    intent,
                    language,
                    "refund_id",
                    ENTITY_VALUES["refund_id"],
                )
            )

    passed = 0
    failed = []

    for number, (
        intent,
        language,
        entity_type,
        entity_value,
    ) in enumerate(tests, 1):

        chatbot = new_chatbot(bot)

        first = chatbot.process_message(
            TEST_CASES[intent][language]
        )

        first_ok = first.get("intent") == intent

        second = chatbot.process_message(
            entity_value
        )

        second_ok = (
            second.get("intent") == intent
        )

        passed_case = (
            first_ok
            and second_ok
        )

        status = "PASS" if passed_case else "FAIL"

        print(
            f"[{number:03d}/{len(tests):03d}] "
            f"{status:<4} "
            f"{language.upper():<2} "
            f"{intent:<30} "
            f"{entity_type}={entity_value}"
        )

        if not passed_case:

            print(
                f"      First:  "
                f"{first.get('intent')} | "
                f"{first.get('response')}"
            )

            print(
                f"      Second: "
                f"{second.get('intent')} | "
                f"{second.get('response')}"
            )

            failed.append(
                {
                    "language": language,
                    "intent": intent,
                    "entity": entity_value,
                    "first_intent": first.get("intent"),
                    "second_intent": second.get("intent"),
                }
            )

        else:
            passed += 1

    return passed, failed, len(tests)


# ============================================================
# CONFIRMATION TESTS
# ============================================================

def run_confirmation_tests(bot):
    """
    Test full:

        user request
             ↓
        entity
             ↓
        confirmation
             ↓
        workflow success

    for every confirmation intent in every language.
    """

    print()
    print("=" * 90)
    print("3. MULTI-TURN CONFIRMATION TESTS")
    print("=" * 90)

    tests = []

    for intent in sorted(CONFIRMATION_INTENTS):
        for language in LANGUAGES:
            tests.append(
                (
                    intent,
                    language,
                )
            )

    passed = 0
    failed = []

    for number, (
        intent,
        language,
    ) in enumerate(tests, 1):

        chatbot = new_chatbot(bot)

        # ----------------------------------------------------
        # STEP 1: Initial request
        # ----------------------------------------------------

        first = chatbot.process_message(
            TEST_CASES[intent][language]
        )

        first_ok = (
            first.get("intent")
            == intent
        )

        # ----------------------------------------------------
        # STEP 2: Entity where required
        # ----------------------------------------------------

        if intent == "delete_account":

            second = first

        else:

            second = chatbot.process_message(
                ENTITY_VALUES["order_id"]
            )

        second_ok = (
            second.get("intent")
            == intent
        )

        confirmation_pending = (
            chatbot.state.is_awaiting_confirmation()
        )

        # ----------------------------------------------------
        # STEP 3: Confirmation
        # ----------------------------------------------------

        third = chatbot.process_message(
            CONFIRMATION_WORDS[language]
        )

        final_ok = (
            third.get("intent")
            == intent
        )

        workflow_success = (
            third.get("workflow_status")
            == "success"
        )

        passed_case = (
            first_ok
            and second_ok
            and confirmation_pending
            and final_ok
            and workflow_success
        )

        status = "PASS" if passed_case else "FAIL"

        print(
            f"[{number:03d}/{len(tests):03d}] "
            f"{status:<4} "
            f"{language.upper():<2} "
            f"{intent:<30} "
            f"workflow={third.get('workflow_status')}"
        )

        if not passed_case:

            print(
                f"      STEP 1: "
                f"{first.get('intent')} | "
                f"{first.get('response')}"
            )

            print(
                f"      STEP 2: "
                f"{second.get('intent')} | "
                f"{second.get('response')}"
            )

            print(
                f"      STEP 3: "
                f"{third.get('intent')} | "
                f"{third.get('response')}"
            )

            print(
                f"      Confirmation pending: "
                f"{confirmation_pending}"
            )

            failed.append(
                {
                    "language": language,
                    "intent": intent,
                    "first": first,
                    "second": second,
                    "third": third,
                }
            )

        else:
            passed += 1

    return passed, failed, len(tests)


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    classification,
    entities,
    confirmations,
):
    print()
    print("=" * 90)
    print("FINAL TEST SUMMARY")
    print("=" * 90)

    class_passed, class_failed, class_total = classification

    entity_passed, entity_failed, entity_total = entities

    confirm_passed, confirm_failed, confirm_total = confirmations

    total_passed = (
        class_passed
        + entity_passed
        + confirm_passed
    )

    total_failed = (
        len(class_failed)
        + len(entity_failed)
        + len(confirm_failed)
    )

    total = (
        class_total
        + entity_total
        + confirm_total
    )

    print()
    print(
        f"Intents:                    {len(TEST_CASES)}"
    )

    print(
        f"Languages:                  {len(LANGUAGES)}"
    )

    print()
    print(
        f"Classification tests:       "
        f"{class_passed}/{class_total} passed"
    )

    print(
        f"Entity tests:               "
        f"{entity_passed}/{entity_total} passed"
    )

    print(
        f"Confirmation tests:         "
        f"{confirm_passed}/{confirm_total} passed"
    )

    print()
    print(
        f"TOTAL TESTS:                {total}"
    )

    print(
        f"TOTAL PASSED:              {total_passed}"
    )

    print(
        f"TOTAL FAILED:              {total_failed}"
    )

    print()

    if total_failed == 0:

        print("=" * 90)
        print("RESULT: ALL TESTS PASSED")
        print("=" * 90)

        return 0

    print("=" * 90)
    print("RESULT: SOME TESTS FAILED")
    print("=" * 90)

    print()

    if class_failed:

        print("CLASSIFICATION FAILURES:")

        for item in class_failed:

            print(
                f"  {item['language'].upper()} | "
                f"expected={item['expected']} | "
                f"actual={item['actual']}"
            )

    if entity_failed:

        print()

        print("ENTITY FAILURES:")

        for item in entity_failed:

            print(
                f"  {item['language'].upper()} | "
                f"{item['intent']} | "
                f"entity={item['entity']}"
            )

    if confirm_failed:

        print()

        print("CONFIRMATION FAILURES:")

        for item in confirm_failed:

            print(
                f"  {item['language'].upper()} | "
                f"{item['intent']} | "
                f"final_intent="
                f"{item['third'].get('intent')} | "
                f"workflow="
                f"{item['third'].get('workflow_status')}"
            )

    return 1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("MULTILINGUAL CUSTOMER SUPPORT CHATBOT")
    print("ALL LANGUAGES / ALL INTENTS TEST SUITE")
    print("=" * 90)

    print()
    print("Languages:")

    for language in LANGUAGES:
        print(
            f"  {language} = "
            f"{LANGUAGE_NAMES[language]}"
        )

    print()
    print(
        f"Intents: {len(TEST_CASES)}"
    )

    print(
        f"Language combinations: "
        f"{len(TEST_CASES) * len(LANGUAGES)}"
    )

    print()
    print("Loading model...")

    chatbot = create_chatbot()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # 1. Classification
    # --------------------------------------------------------

    classification = run_classification_tests(
        chatbot
    )

    # --------------------------------------------------------
    # 2. Entities
    # --------------------------------------------------------

    entities = run_entity_tests(
        chatbot
    )

    # --------------------------------------------------------
    # 3. Confirmations
    # --------------------------------------------------------

    confirmations = run_confirmation_tests(
        chatbot
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    return print_summary(
        classification,
        entities,
        confirmations,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())