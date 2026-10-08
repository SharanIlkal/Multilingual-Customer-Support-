import re
from src.models.predict import predict

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.50

class IntentRouter:
    def __init__(
        self,
        model,
        vocab,
        id_to_label
    ):
        self.model = model
        self.vocab = vocab
        self.id_to_label = id_to_label

    OVERRIDES = {
        "track_order": [
            "track my order",
            "track order",
            "where is my order",
            "order status",
            "track shipment",
            "track package",

            "मेरा ऑर्डर ट्रैक",
            "ऑर्डर ट्रैक",
            "मेरा ऑर्डर",

            "ಆರ್ಡರ್ ಟ್ರ್ಯಾಕ್",
            "ನನ್ನ ಆರ್ಡರ್",

            "माझा ऑर्डर",
            "ऑर्डर ट्रॅक",

            "எனது ஆர்டர்",
            "ஆர்டரை கண்காணிக்க",

            "నా ఆర్డర్",
            "ఆర్డర్‌ను ట్రాక్",
        ],

        "track_refund": [
            "track refund",
            "refund status",
            "refund tracking",
            "status of my refund",

            "रिफंड ट्रैक",
            "रिफंड की स्थिति",
            "मेरा रिफंड",

            "ರಿಫಂಡ್ ಟ್ರ್ಯಾಕ್",
            "ರಿಫಂಡ್ ಸ್ಥಿತಿ",
            "ನನ್ನ ರಿಫಂಡ್",

            "रिफंड ट्रॅक",
            "रिफंडची स्थिती",
            "माझा रिफंड",

            "ரீஃபண்டை கண்காணிக்க",
            "ரீஃபண்ட் நிலை",
            "எனது ரீஃபண்ட்",

            "రీఫండ్‌ను ట్రాక్",
            "రీఫండ్ స్థితి",
            "నా రీఫండ్",
        ],

        "get_refund": [
            "get a refund",
            "request a refund",
            "want a refund",
            "need a refund",
            "refund for my order",
            "refund my order",

            "रिफंड चाहिए",
            "रिफंड लेना",
            "रिफंड का अनुरोध",

            "ರಿಫಂಡ್ ಬೇಕು",
            "ರಿಫಂಡ್ ಪಡೆಯ",

            "रिफंड हवा",
            "रिफंड घ्यायचा",
            "रिफंडची विनंती",

            "ரீஃபண்ட் வேண்டும்",
            "ரீஃபண்ட் பெற",

            "రీఫండ్ కావాలి",
            "రీఫండ్ పొంద",
        ],

        "cancel_order": [
            "cancel order",
            "cancel my order",
            "want to cancel my order",
            "cancel it",

            "ऑर्डर रद्द",
            "ऑर्डर कैंसल",
            "रद्द करना",

            "ಆರ್ಡರ್ ರದ್ದು",
            "ನನ್ನ ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
            "ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
            "ನನ್ನ ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
            "ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
            "ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್",

            "ऑर्डर रद्द",
            "ऑर्डर कॅन्सल",
            "रद्द करायची",

            "ஆர்டரை ரத்து",
            "ஆர்டர் கேன்சல்",

            "ఆర్డర్ రద్దు",
            "ఆర్డర్ క్యాన్సల్",
        
            "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",],

        "change_order": [
            "ನನ್ನ ಆರ್ಡರ್ ಬದಲಾಯಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ನನ್ನ ಆರ್ಡರ್‌ನಲ್ಲಿ ಬದಲಾವಣೆ ಮಾಡಬೇಕು",
            "ನನ್ನ ಆರ್ಡರ್‌ನಲ್ಲಿ ಬದಲಾವಣೆ ಬೇಕು",
            "माझा ऑर्डर बदलायचा आहे",
            "माझ्या ऑर्डरमध्ये बदल हवा आहे",
            "ना ऑर्डर बदलायचा आहे",
            "ना ऑर्डर बदलायचा आहे",
            "నా ఆర్డర్ మార్చాలని ఉంది",
            "నా ఆర్డర్‌లో మార్పు చేయాలి",
            "నా ఆర్డర్‌లో మార్పు కావాలి",
            "change my order",
            "modify my order",
            "edit my order",
            "change order",

            "ऑर्डर बदल",
            "ऑर्डर में बदलाव",
            "ऑर्डर संशोधित",

            "ಆರ್ಡರ್ ಬದಲಾಯ",
            "ಆರ್ಡರ್ ಬದಲಾವಣೆ",

            "ऑर्डरमध्ये बदल",
            "माझ्या ऑर्डरमध्ये बदल",

            "ஆர்டரை மாற்ற",
            "ஆர்டரில் மாற்றம்",

            "ఆర్డర్ మార్చ",
            "ఆర్డర్‌లో మార్పు",
        ],

        "delivery_period": [
            "how long is delivery",
            "how long does shipping take",
            "how long will shipping take",
            "how long does it take",
            "how long will delivery take",
            "how long does delivery take",
            "delivery time",
            "delivery period",
            "when will my order arrive",
            "when will it arrive",

            "कितने दिन डिलीवरी",
            "डिलीवरी में कितना समय",
            "कब आएगा",

            "ಎಷ್ಟು ದಿನ ಡೆಲಿವರಿ",
            "ಡೆಲಿವರಿ ಸಮಯ",
            "ಯಾವಾಗ ಬರುತ್ತದೆ",

            "डिलिव्हरीला किती वेळ",
            "कधी येईल",

            "எவ்வளவு நேரம் டெலிவரி",
            "எப்போது வரும்",

            "ఎంత సమయం డెలివరీ",
            "ఎప్పుడు వస్తుంది",
        ],

        "delivery_options": [
            "delivery options",
            "delivery option",
            "shipping options",
            "shipping option",
            "express delivery",
            "standard delivery",

            "डिलीवरी विकल्प",
            "शिपिंग विकल्प",

            "ಡೆಲಿವರಿ ಆಯ್ಕೆ",
            "ಶಿಪ್ಪಿಂಗ್ ಆಯ್ಕೆ",

            "डिलिव्हरी पर्याय",
            "शिपिंग पर्याय",

            "டெலிவரி விருப்பம்",
            "ஷிப்பிங் விருப்பம்",

            "డెలివరీ ఎంపిక",
            "షిప్పింగ్ ఎంపిక",
        ],

        "payment_issue": [
            "எனது பேமெண்ட் பிரச்சனை",
            "பேமெண்ட் வேலை செய்யவில்லை",
            "என்னால் பணம் செலுத்த முடியவில்லை",
            "payment failed",
            "payment problem",
            "payment issue",
            "cannot make payment",
            "payment not working",

            "भुगतान समस्या",
            "पेमेंट समस्या",

            "ಪಾವತಿ ಸಮಸ್ಯೆ",

            "पेमेंटची समस्या",

            "பேமெண்ட் பிரச்சனை",

            "చెల్లింపు సమస్య",
        ],

        "check_payment_methods": [
            "payment methods",
            "payment options",
            "how can i pay",
            "which payment methods",

            "भुगतान तरीके",
            "पेमेंट पद्धती",

            "ಪಾವತಿ ವಿಧಾನ",

            "पेमेंट पद्धत",

            "பேமெண்ட் முறைகள்",

            "చెల్లింపు పద్ధతులు",
        ],

        "check_invoice": [
            "check invoice",
            "is invoice available",
            "invoice available",
            "invoice status",

            "इनवॉइस उपलब्ध",
            "बिल उपलब्ध",

            "ಇನ್ವಾಯ್ಸ್ ಲಭ್ಯ",

            "इनव्हॉइस उपलब्ध",

            "இன்வாய்ஸ் கிடைக்க",

            "ఇన్వాయిస్ అందుబాటులో",
        ],

        "get_invoice": [
            "எனக்கு இன்வாய்ஸ் வேண்டும்",
            "எனது இன்வாய்ஸ் வேண்டும்",
            "இன்வாய்ஸை பதிவிறக்க வேண்டும்",
            "get invoice",
            "download invoice",
            "send invoice",
            "need invoice",
            "invoice for my order",

            "इनवॉइस चाहिए",
            "इनवॉइस प्राप्त",

            "ಇನ್ವಾಯ್ಸ್ ಬೇಕು",

            "इनव्हॉइस पाहिजे",

            "இன்வாய்ஸ் வேண்டும்",

            "ఇన్వాయిస్ కావాలి",
        ],

        "change_shipping_address": [
            "change shipping address",
            "change delivery address",
            "update shipping address",
            "change my address",

            "शिपिंग पता बदल",
            "डिलीवरी पता बदल",

            "ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸ ಬದಲಾಯ",

            "डिलिव्हरी पत्ता बदल",

            "ஷிப்பிங் முகவரியை மாற்ற",

            "షిప్పింగ్ చిరునామా మార్చ",
        ],

        "set_up_shipping_address": [
            "i want to set up a shipping address",
            "i need to set up a shipping address",
            "i want to add a shipping address",
            "i need to add a shipping address",
            "how do i add a shipping address",
            "how can i add a shipping address",
            "set up my shipping address",
            "setup my shipping address",
            "ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸ ಸೇರಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ಹೊಸ ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸ ಸೇರಿಸಬೇಕು",
            "मुझे शिपिंग पता जोड़ना है",
            "मुझे नया शिपिंग पता जोड़ना है",
            "मला शिपिंग पत्ता जोडायचा आहे",
            "नवीन शिपिंग पत्ता जोडायचा आहे",
            "எனது ஷிப்பிங் முகவரியை சேர்க்க வேண்டும்",
            "புதிய ஷிப்பிங் முகவரி சேர்க்க வேண்டும்",
            "నా షిప్పింగ్ చిరునామా జోడించాలి",
            "కొత్త షిప్పింగ్ చిరునామా జోడించాలి",
            "add shipping address",
            "set shipping address",
            "add delivery address",
            "save shipping address",

            "शिपिंग पता जोड़",
            "पता जोड़ना",

            "ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸ ಸೇರಿಸ",

            "पत्ता जोडणे",

            "ஷிப்பிங் முகவரி சேர்க்க",

            "షిప్పింగ్ చిరునామా జోడించ",
        ],

        "recover_password": [
            "எனது கடவுச்சொல்லை மறந்துவிட்டேன்",
            "என் கடவுச்சொல்லை மறந்துவிட்டேன்",
            "கடவுச்சொல் மறந்துவிட்டது",
            "எனது கடவுச்சொல் மறந்துவிட்டது",
            "forgot password",
            "reset password",
            "recover password",
            "password reset",

            "पासवर्ड भूल",
            "पासवर्ड रीसेट",

            "ಪಾಸ್‌ವರ್ಡ್ ಮರೆತು",

            "पासवर्ड विसरलो",

            "கடவுச்சொல் மறந்துவிட்டேன்",

            "పాస్‌వర్డ్ మర్చిపోయాను",
        ],

        "create_account": [
            "create account",
            "new account",
            "sign up",
            "register account",

            "नया अकाउंट",
            "खाता बनाना",

            "ಹೊಸ ಖಾತೆ",

            "नवीन खाते",

            "புதிய கணக்கு",

            "కొత్త ఖాతా",
        ],

        "delete_account": [
            "delete account",
            "close account",
            "remove account",

            "अकाउंट डिलीट",
            "खाता हटाना",

            "ಖಾತೆ ಅಳಿಸಿ",

            "खाते हटवायचे",

            "கணக்கை நீக்க",

            "ఖాతాను తొలగించ",
        ],

        "edit_account": [
            "edit account",
            "update account",
            "change account details",
            "update profile",

            "अकाउंट एडिट",
            "खाता अपडेट",

            "ಖಾತೆ ಮಾಹಿತಿ ಬದಲಾಯ",

            "खाते अपडेट",

            "கணக்கு தகவலை மாற்ற",

            "ఖాతా సమాచారాన్ని మార్చ",
        ],

        "switch_account": [
            "switch account",
            "change account",
            "use another account",

            "दूसरे अकाउंट",
            "खाता बदलना",

            "ಬೇರೆ ಖಾತೆ",

            "दुसरे खाते",

            "வேறு கணக்கு",

            "మరొక ఖాతా",
        ],

        "check_cancellation_fee": [
            "cancellation fee",
            "cancel fee",
            "cancellation charge",

            "रद्द शुल्क",
            "कैंसलेशन शुल्क",

            "ರದ್ದು ಶುಲ್ಕ",

            "रद्द शुल्क",

            "ரத்து கட்டணம்",

            "రద్దు రుసుము",
        ],

        "contact_human_agent": [
            "talk to human",
            "human agent",
            "speak to agent",
            "customer representative",

            "मानव प्रतिनिधि",
            "सहायता प्रतिनिधि",

            "ಮಾನವ ಪ್ರತಿನಿಧಿ",

            "सहाय्य प्रतिनिधी",

            "மனித பிரதிநிதி",

            "మానవ ప్రతినిధి",
        ],

        "contact_customer_service": [
            "కస్టమర్ సర్వీస్‌ను సంప్రదించాలి",
            "కస్టమర్ సర్వీస్‌ని సంప్రదించాలి",
            "నాకు కస్టమర్ సేవ కావాలి",
            "customer service",
            "contact support",
            "contact customer service",
            "support team",

            "ग्राहक सेवा",
            "कस्टमर सर्विस",

            "ಗ್ರಾಹಕ ಸೇವೆ",

            "ग्राहक सेवा",

            "வாடிக்கையாளர் சேவை",

            "కస్టమర్ సేవ",
        ],

        "complaint": [
            "make a complaint",
            "file a complaint",
            "customer complaint",
            "complaint",

            "शिकायत",
            "तक्रार",

            "ದೂರು",

            "புகார்",

            "ఫిర్యాదు",
        ],

        "review": [
            "मैं समीक्षा देना चाहता हूं",
            "मुझे समीक्षा देनी है",
            "मैं रिव्यू देना चाहता हूं",
            "நான் மதிப்புரை வழங்க வேண்டும்",
            "நான் ஒரு மதிப்புரை வழங்க விரும்புகிறேன்",
            "விமர்சனம் எழுத வேண்டும்",
            "leave a review",
            "write a review",
            "give feedback",
            "review",

            "समीक्षा",
            "अभिप्राय",

            "ವಿಮರ್ಶೆ",

            "पुनरावलोकन",

            "மதிப்புரை",

            "సమీక్ష",
        ],

        "newsletter_subscription": [
            "நியூஸ்லெட்டர் சந்தா",
            "நியூஸ்லெட்டரில் சேர வேண்டும்",
            "நியூஸ்லெட்டருக்கு பதிவு செய்ய வேண்டும்",
            "செய்திமடலில் சேர வேண்டும்",
            "newsletter",
            "subscribe newsletter",
            "unsubscribe newsletter",
            "newsletter subscription",

            "न्यूज़लेटर",
            "न्यूजलेटर",

            "ಸುದ್ದಿಪತ್ರ",

            "वृत्तपत्र सदस्यता",

            "நியூஸ்லெட்டர்",

            "వార్తాలేఖ",
        ],

        "registration_problems": [
            "i have a problem registering",
            "i am having trouble registering",
            "i cannot complete registration",
            "i cannot complete the registration",
            "registration problem",
            "cannot register",
            "registration failed",
            "sign up problem",

            "पंजीकरण समस्या",
            "रजिस्ट्रेशन समस्या",

            "ನೋಂದಣಿ ಸಮಸ್ಯೆ",

            "नोंदणी समस्या",

            "பதிவு பிரச்சனை",

            "నమోదు సమస్య",
        ],

        "place_order": [
            "मुझे ऑर्डर करना है",
            "मैं ऑर्डर करना चाहता हूं",
            "मुझे कुछ खरीदना है",
            "place an order",
            "place order",
            "buy something",
            "make a purchase",

            "ऑर्डर करना",
            "खरीदना",

            "ಆರ್ಡರ್ ಮಾಡ",

            "खरेदी कर",

            "ஆர்டர் செய்ய",

            "ఆర్డర్ చేయ",
        ],
    }

    OVERRIDE_PRIORITY = [
        "cancel_order", "change_order", "set_up_shipping_address",
        "change_shipping_address", "place_order", "get_invoice",
        "check_invoice", "payment_issue", "check_refund_policy",
        "recover_password", "registration_problems",
        "contact_customer_service", "contact_human_agent",
        "delivery_period", "delivery_options", "review",
        "newsletter_subscription", "track_refund", "get_refund",
        "track_order",
    ]

    EXACT_OVERRIDES = {
        "registration_problems": [
            "i have a registration problem", "registration problem",
            "registration failed", "i cannot register", "i can't register",
            "sign up problem",
        ],
        "place_order": [
            "ऑर्डर करना है", "ऑर्डर करना", "खरीदना है",
            "मैं ऑर्डर देना चाहता हूँ",
        ],
        "change_order": [
            "ನನ್ನ ಆರ್ಡರ್ ಬದಲಾಯಿಸಬೇಕು", "ಆರ್ಡರ್ ಬದಲಾಯಿಸಬೇಕು",
            "माझ्या ऑर्डरमध्ये बदल करायचा आहे", "माझ्या ऑर्डरमध्ये बदल",
            "ऑर्डरमध्ये बदल करायचा आहे", "నా ఆర్డర్ మార్చాలి",
            "ఆర్డర్ మార్చాలి", "ఆర్డర్‌లో మార్పు చేయాలి",
            "\u0c28\u0c3e \u0c06\u0c30\u0c4d\u0c21\u0c30\u0c4d\u200c\u0c28\u0c41 \u0c2e\u0c3e\u0c30\u0c4d\u0c1a\u0c3e\u0c32\u0c28\u0c41\u0c15\u0c41\u0c02\u0c1f\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c28\u0c41",
        ],
        "set_up_shipping_address": [
            "add shipping address", "set shipping address", "add delivery address",
            "save shipping address", "i want to add my shipping address",
            "i want to set my shipping address", "i want to add my delivery address",
            "i want to save my shipping address", "शिपिंग पता जोड़ना है",
            "पता जोड़ना है", "पत्ता जोडायचा आहे", "शिपिंग पत्ता जोडायचा आहे",
            "ஷிப்பிங் முகவரி சேர்க்க வேண்டும்", "ஷிப்பிங் முகவரியை சேர்க்க வேண்டும்",
            "షిప్పింగ్ చిరునామా జోడించాలి", "షిప్పింగ్ చిరునామాను జోడించాలి",
            "ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸ ಸೇರಿಸಬೇಕು",
            "ನಾನು ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸವನ್ನು ಸೇರಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "நான் ஷிப்பிங் முகவரியைச் சேர்க்க விரும்புகிறேன்",
            "నేను షిప్పింగ్ చిరునామాను జోడించాలనుకుంటున్నాను",
        ],
        "contact_customer_service": [
            "customer service", "contact customer service", "contact support",
            "support team", "வாடிக்கையாளர் சேவை", "கஸ்டமர் சேவை",
            "కస్టమర్ సేవ", "కస్టమర్ సర్వీస్",
            "నేను ಕಸ್ಟಮರ್ ಸರ್ವೀಸ್ ಅನ್ನು సంప్రదించాలనుకుంటున్నాను",
            "\u0c28\u0c47\u0c28\u0c41 \u0c15\u0cb8\u0ccd\u0c9f\u0cae\u0c30\u0c4d \u0cb8\u0cb0\u0ccd\u0cb5\u0c40\u0cb8\u0ccd \u0c85\u0ca8\u0ccd\u0ca8\u0cc1 \u0c38\u0c02\u0c2a\u0c4d\u0c30\u0c26\u0c3f\u0c02\u0c1a\u0c3e\u0c32\u0c28\u0c41\u0c15\u0c41\u0c02\u0c1f\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c28\u0c41",
            "\u0c28\u0c47\u0c28\u0c41 \u0c15\u0cb8\u0ccd\u0c9f\u0cae\u0c30\u0c4d \u0cb8\u0cb0\u0ccd\u0cb5\u0c40\u0c38\u0c4d \u0c85\u0ca8\u0ccd\u0ca8\u0cc1 \u0c38\u0c02\u0c2a\u0c4d\u0c30\u0c26\u0c3f\u0c02\u0c1a\u0c3e\u0c32\u0c28\u0c41\u0c15\u0c41\u0c02\u0c1f\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c28\u0c41",
        ],
        "recover_password": [
            "கடவுச்சொல்லை மறந்துவிட்டேன்", "கடவுச்சொல் மறந்துவிட்டேன்",
            "கடவுச்சொல்லை மீட்டமைக்க வேண்டும்", "கடவுச்சொல் மீட்டமைக்க வேண்டும்",
        ],
        "get_invoice": [
            "இன்வாய்ஸ் வேண்டும்", "இன்வாய்ஸை பெற வேண்டும்",
            "இன்வாய்ஸ் பெற வேண்டும்", "இன்வாய்ஸ் பதிவிறக்க வேண்டும்",
            "எனது இன்வாய்ஸை பெற விரும்புகிறேன்",
        ],
        "payment_issue": [
            "பேமெண்ட் பிரச்சனை", "பணம் செலுத்துவதில் பிரச்சனை",
            "பணம் செலுத்த முடியவில்லை",
            "எனது பேமெண்ட்டில் சிக்கல் உள்ளது",
        ],
        "check_refund_policy": [
            "ರಿಫಂಡ್ ನೀತಿ", "ರಿಫಂಡ್ ನೀತಿ ಏನು", "ರಿಫಂಡ್ ಪಾಲಿಸಿ",
            "ரீஃபண்ட் கொள்கை", "ரீஃபண்ட் கொள்கை என்ன", "ரீஃபண்ட் பாலிசி",
        ],
    }

    def _normalize(self, message):
        return re.sub(r"\s+", " ", message.lower().strip())

    def _override_result(self, intent, category, confidence, predictions, source):
        return {
            "action": "handle_intent", "intent": intent, "category": category,
            "confidence": max(confidence, 0.99),
            "top_predictions": predictions, "routing_source": source,
        }

    def classify(self, message):
        predictions, _ = predict(
            message,
            self.model,
            self.vocab,
            self.id_to_label,
            top_k=3
        )

        best = predictions[0]
        intent = best["intent"]
        category = best["category"]
        confidence = float(best["confidence"])
        normalized = self._normalize(message)

        for override_intent, phrases in self.EXACT_OVERRIDES.items():
            if any(normalized == phrase.lower().strip() for phrase in phrases):
                return self._override_result(override_intent, category, confidence, predictions, "multilingual_exact_override")

        matches = []
        broad_order_phrases = {
            "मेरा ऑर्डर", "ನನ್ನ ಆರ್ಡರ್", "माझा ऑर्डर", "எனது ஆர்டர்", "నా ఆర్డర్"
        }

        for priority, override_intent in enumerate(self.OVERRIDE_PRIORITY):
            for phrase in self.OVERRIDES.get(override_intent, []):
                phrase = self._normalize(phrase)
                if not phrase:
                    continue
                if phrase in broad_order_phrases:
                    matched = normalized == phrase
                else:
                    matched = phrase in normalized
                if matched:
                    matches.append((len(phrase), -priority, override_intent))

        if matches:
            _, _, override_intent = max(matches)
            return self._override_result(
                override_intent, category, confidence, predictions,
                "multilingual_override"
            )

        if confidence >= HIGH_CONFIDENCE:
            action = "handle_intent"

        elif confidence >= MEDIUM_CONFIDENCE:
            action = "clarify"

        else:
            action = "human_escalation"

        return {
            "action": action,
            "intent": intent,
            "category": category,
            "confidence": confidence,
            "top_predictions": predictions,
            "routing_source": "model",
        }
