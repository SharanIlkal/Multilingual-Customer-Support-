from .response_templates import get_response

class ResponseGenerator:
    LANGUAGES = {
        "en",
        "hi",
        "kn",
        "mr",
        "ta",
        "te",
    }

    STATIC = {
        "place_order": {
            "en": "You can place an order by selecting a product and completing checkout.",
            "hi": "आप उत्पाद चुनकर और चेकआउट पूरा करके ऑर्डर कर सकते हैं।",
            "kn": "ಉತ್ಪನ್ನವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ ಚೆಕ್‌ಔಟ್ ಪೂರ್ಣಗೊಳಿಸುವ ಮೂಲಕ ಆರ್ಡರ್ ಮಾಡಬಹುದು.",
            "mr": "उत्पादन निवडून आणि चेकआउट पूर्ण करून तुम्ही ऑर्डर देऊ शकता.",
            "ta": "ஒரு பொருளைத் தேர்ந்தெடுத்து செக்அவுட்டை முடித்து ஆர்டர் செய்யலாம்.",
            "te": "ఉత్పత్తిని ఎంచుకుని చెక్అవుట్ పూర్తి చేసి ఆర్డర్ చేయవచ్చు.",
        },

        "set_up_shipping_address": {
            "en": "You can add a shipping address from your account before placing an order.",
            "hi": "ऑर्डर देने से पहले आप अपने अकाउंट में शिपिंग पता जोड़ सकते हैं।",
            "kn": "ಆರ್ಡರ್ ಮಾಡುವ ಮೊದಲು ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸವನ್ನು ಸೇರಿಸಬಹುದು.",
            "mr": "ऑर्डर देण्यापूर्वी तुम्ही तुमच्या खात्यात शिपिंग पत्ता जोडू शकता.",
            "ta": "ஆர்டர் செய்வதற்கு முன் உங்கள் கணக்கில் ஷிப்பிங் முகவரியைச் சேர்க்கலாம்.",
            "te": "ఆర్డర్ చేసే ముందు మీ ఖాతాలో షిప్పింగ్ చిరునామాను జోడించవచ్చు.",
        },

        "delivery_options": {
            "en": "Available delivery options may include standard and express delivery depending on your location and order.",
            "hi": "आपके स्थान और ऑर्डर के आधार पर मानक और एक्सप्रेस डिलीवरी विकल्प उपलब्ध हो सकते हैं।",
            "kn": "ನಿಮ್ಮ ಸ್ಥಳ ಮತ್ತು ಆರ್ಡರ್‌ಗೆ ಅನುಗುಣವಾಗಿ ಸ್ಟ್ಯಾಂಡರ್ಡ್ ಮತ್ತು ಎಕ್ಸ್‌ಪ್ರೆಸ್ ಡೆಲಿವರಿ ಆಯ್ಕೆಗಳು ಲಭ್ಯವಿರಬಹುದು.",
            "mr": "तुमच्या ठिकाण आणि ऑर्डरनुसार स्टँडर्ड आणि एक्सप्रेस डिलिव्हरी पर्याय उपलब्ध असू शकतात.",
            "ta": "உங்கள் இருப்பிடம் மற்றும் ஆர்டரைப் பொறுத்து ஸ்டாண்டர்ட் மற்றும் எக்ஸ்பிரஸ் டெலிவரி விருப்பங்கள் இருக்கலாம்.",
            "te": "మీ స్థానం మరియు ఆర్డర్ ఆధారంగా స్టాండర్డ్ మరియు ఎక్స్‌ప్రెస్ డెలివరీ ఎంపికలు అందుబాటులో ఉండవచ్చు.",
        },

        "delivery_period": {
            "en": "Standard delivery usually takes a few business days. The exact delivery date is shown during checkout.",
            "hi": "मानक डिलीवरी में आमतौर पर कुछ कार्यदिवस लगते हैं। सटीक डिलीवरी तारीख चेकआउट के दौरान दिखाई जाती है।",
            "kn": "ಸ್ಟ್ಯಾಂಡರ್ಡ್ ಡೆಲಿವರಿಗೆ ಸಾಮಾನ್ಯವಾಗಿ ಕೆಲವು ಕೆಲಸದ ದಿನಗಳು ಬೇಕಾಗುತ್ತವೆ. ನಿಖರವಾದ ಡೆಲಿವರಿ ದಿನಾಂಕ ಚೆಕ್‌ಔಟ್‌ನಲ್ಲಿ ತೋರಿಸಲಾಗುತ್ತದೆ.",
            "mr": "स्टँडर्ड डिलिव्हरीला साधारण काही कामकाजाचे दिवस लागतात. अचूक डिलिव्हरी तारीख चेकआउटदरम्यान दिसते.",
            "ta": "ஸ்டாண்டர்ட் டெலிவரிக்கு பொதுவாக சில வேலை நாட்கள் ஆகும். சரியான டெலிவரி தேதி செக்அவுட்டில் காட்டப்படும்.",
            "te": "స్టాండర్డ్ డెలివరీకి సాధారణంగా కొన్ని పని రోజులు పడతాయి. ఖచ్చితమైన డెలివరీ తేదీ చెక్అవుట్‌లో చూపబడుతుంది.",
        },

        "check_refund_policy": {
            "en": "Refunds are available for eligible orders. Eligibility depends on the order status and purchase conditions.",
            "hi": "योग्य ऑर्डर के लिए रिफंड उपलब्ध है। पात्रता ऑर्डर की स्थिति और खरीद की शर्तों पर निर्भर करती है।",
            "kn": "ಅರ್ಹ ಆರ್ಡರ್‌ಗಳಿಗೆ ರಿಫಂಡ್ ಲಭ್ಯವಿದೆ. ಅರ್ಹತೆ ಆರ್ಡರ್ ಸ್ಥಿತಿ ಮತ್ತು ಖರೀದಿ ನಿಯಮಗಳ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ.",
            "mr": "पात्र ऑर्डरसाठी रिफंड उपलब्ध आहे. पात्रता ऑर्डरची स्थिती आणि खरेदीच्या अटींवर अवलंबून असते.",
            "ta": "தகுதியான ஆர்டர்களுக்கு ரீஃபண்ட் கிடைக்கும். தகுதி ஆர்டர் நிலை மற்றும் வாங்கிய நிபந்தனைகளைப் பொறுத்தது.",
            "te": "అర్హత ఉన్న ఆర్డర్లకు రీఫండ్ అందుబాటులో ఉంటుంది. అర్హత ఆర్డర్ స్థితి మరియు కొనుగోలు నిబంధనలపై ఆధారపడి ఉంటుంది.",
        },

        "payment_issue": {
            "en": "I'm sorry you're having trouble with your payment. Please check your payment details and try again.",
            "hi": "आपको भुगतान में समस्या हो रही है, इसके लिए खेद है। कृपया भुगतान विवरण जांचकर फिर से प्रयास करें।",
            "kn": "ಪಾವತಿಯಲ್ಲಿ ಸಮಸ್ಯೆಯಾಗಿರುವುದಕ್ಕೆ ಕ್ಷಮಿಸಿ. ನಿಮ್ಮ ಪಾವತಿ ವಿವರಗಳನ್ನು ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            "mr": "तुम्हाला पेमेंटमध्ये समस्या येत असल्याबद्दल क्षमस्व. पेमेंट तपशील तपासून पुन्हा प्रयत्न करा.",
            "ta": "பேமெண்ட்டில் சிக்கல் ஏற்பட்டதற்கு மன்னிக்கவும். உங்கள் பேமெண்ட் விவரங்களைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும்.",
            "te": "చెల్లింపులో సమస్య వచ్చినందుకు క్షమించండి. మీ చెల్లింపు వివరాలను తనిఖీ చేసి మళ్లీ ప్రయత్నించండి.",
        },

        "check_payment_methods": {
            "en": "Available payment methods may include cards, online banking and other supported options shown during checkout.",
            "hi": "उपलब्ध भुगतान विधियों में कार्ड, ऑनलाइन बैंकिंग और चेकआउट में दिखाए गए अन्य विकल्प शामिल हो सकते हैं।",
            "kn": "ಲಭ್ಯವಿರುವ ಪಾವತಿ ವಿಧಾನಗಳಲ್ಲಿ ಕಾರ್ಡ್‌ಗಳು, ಆನ್‌ಲೈನ್ ಬ್ಯಾಂಕಿಂಗ್ ಮತ್ತು ಚೆಕ್‌ಔಟ್‌ನಲ್ಲಿ ತೋರಿಸಲಾದ ಇತರ ಆಯ್ಕೆಗಳು ಇರಬಹುದು.",
            "mr": "उपलब्ध पेमेंट पद्धतींमध्ये कार्ड, ऑनलाइन बँकिंग आणि चेकआउटमध्ये दाखवलेले इतर पर्याय असू शकतात.",
            "ta": "கிடைக்கும் பேமெண்ட் முறைகளில் கார்டுகள், ஆன்லைன் பேங்கிங் மற்றும் செக்அவுட்டில் காட்டப்படும் பிற விருப்பங்கள் இருக்கலாம்.",
            "te": "అందుబాటులో ఉన్న చెల్లింపు పద్ధతుల్లో కార్డులు, ఆన్‌లైన్ బ్యాంకింగ్ మరియు చెక్అవుట్‌లో చూపించే ఇతర ఎంపికలు ఉండవచ్చు.",
        },

        "recover_password": {
            "en": "You can reset your password using the password reset option. A reset link will be sent to your registered email address.",
            "hi": "आप पासवर्ड रीसेट विकल्प का उपयोग करके अपना पासवर्ड बदल सकते हैं। रीसेट लिंक आपके पंजीकृत ईमेल पर भेजा जाएगा।",
            "kn": "ಪಾಸ್‌ವರ್ಡ್ ರೀಸೆಟ್ ಆಯ್ಕೆಯನ್ನು ಬಳಸಿಕೊಂಡು ನಿಮ್ಮ ಪಾಸ್‌ವರ್ಡ್ ಬದಲಾಯಿಸಬಹುದು. ರೀಸೆಟ್ ಲಿಂಕ್ ನಿಮ್ಮ ನೋಂದಾಯಿತ ಇಮೇಲ್‌ಗೆ ಕಳುಹಿಸಲಾಗುತ್ತದೆ.",
            "mr": "पासवर्ड रीसेट पर्याय वापरून तुम्ही पासवर्ड बदलू शकता. रीसेट लिंक तुमच्या नोंदणीकृत ईमेलवर पाठवला जाईल.",
            "ta": "கடவுச்சொல் மீட்டமைப்பு விருப்பத்தைப் பயன்படுத்தி உங்கள் கடவுச்சொல்லை மாற்றலாம். மீட்டமைப்பு இணைப்பு பதிவு செய்யப்பட்ட மின்னஞ்சலுக்கு அனுப்பப்படும்.",
            "te": "పాస్‌వర్డ్ రీసెట్ ఎంపికను ఉపయోగించి మీ పాస్‌వర్డ్‌ను మార్చవచ్చు. రీసెట్ లింక్ మీ నమోదిత ఇమెయిల్‌కు పంపబడుతుంది.",
        },

        "create_account": {
            "en": "You can create a new account using your email address and completing the registration process.",
            "hi": "आप अपने ईमेल पते का उपयोग करके पंजीकरण प्रक्रिया पूरी करके नया अकाउंट बना सकते हैं।",
            "kn": "ನಿಮ್ಮ ಇಮೇಲ್ ಬಳಸಿ ನೋಂದಣಿ ಪ್ರಕ್ರಿಯೆಯನ್ನು ಪೂರ್ಣಗೊಳಿಸಿ ಹೊಸ ಖಾತೆಯನ್ನು ರಚಿಸಬಹುದು.",
            "mr": "तुमचा ईमेल वापरून नोंदणी प्रक्रिया पूर्ण करून नवीन खाते तयार करू शकता.",
            "ta": "உங்கள் மின்னஞ்சலைப் பயன்படுத்தி பதிவு செயல்முறையை முடித்து புதிய கணக்கை உருவாக்கலாம்.",
            "te": "మీ ఇమెయిల్ ఉపయోగించి నమోదు ప్రక్రియను పూర్తి చేసి కొత్త ఖాతాను సృష్టించవచ్చు.",
        },

        "edit_account": {
            "en": "You can update your name, email address, phone number and other profile details.",
            "hi": "आप अपना नाम, ईमेल, फोन नंबर और अन्य प्रोफाइल जानकारी अपडेट कर सकते हैं।",
            "kn": "ನಿಮ್ಮ ಹೆಸರು, ಇಮೇಲ್, ಫೋನ್ ಸಂಖ್ಯೆ ಮತ್ತು ಇತರ ಪ್ರೊಫೈಲ್ ವಿವರಗಳನ್ನು ನವೀಕರಿಸಬಹುದು.",
            "mr": "तुमचे नाव, ईमेल, फोन नंबर आणि इतर प्रोफाइल तपशील अपडेट करू शकता.",
            "ta": "உங்கள் பெயர், மின்னஞ்சல், தொலைபேசி எண் மற்றும் பிற சுயவிவர விவரங்களைப் புதுப்பிக்கலாம்.",
            "te": "మీ పేరు, ఇమెయిల్, ఫోన్ నంబర్ మరియు ఇతర ప్రొఫైల్ వివరాలను నవీకరించవచ్చు.",
        },

        "switch_account": {
            "en": "You can switch accounts by signing out of the current account and signing in with another account.",
            "hi": "आप वर्तमान अकाउंट से साइन आउट करके दूसरे अकाउंट में साइन इन कर सकते हैं।",
            "kn": "ಪ್ರಸ್ತುತ ಖಾತೆಯಿಂದ ಸೈನ್ ಔಟ್ ಮಾಡಿ ಮತ್ತೊಂದು ಖಾತೆಗೆ ಸೈನ್ ಇನ್ ಮಾಡುವ ಮೂಲಕ ಖಾತೆ ಬದಲಾಯಿಸಬಹುದು.",
            "mr": "सध्याच्या खात्यातून साइन आउट करून दुसऱ्या खात्यात साइन इन करून खाते बदलू शकता.",
            "ta": "தற்போதைய கணக்கிலிருந்து வெளியேறி மற்றொரு கணக்கில் உள்நுழைந்து கணக்கை மாற்றலாம்.",
            "te": "ప్రస్తుత ఖాతా నుండి సైన్ అవుట్ చేసి మరొక ఖాతాలో సైన్ ఇన్ చేయడం ద్వారా ఖాతాను మార్చవచ్చు.",
        },

        "check_cancellation_fee": {
            "en": "Cancellation fees depend on the order status and applicable purchase conditions.",
            "hi": "रद्दीकरण शुल्क ऑर्डर की स्थिति और लागू खरीद शर्तों पर निर्भर करता है।",
            "kn": "ರದ್ದು ಶುಲ್ಕವು ಆರ್ಡರ್ ಸ್ಥಿತಿ ಮತ್ತು ಅನ್ವಯಿಸುವ ಖರೀದಿ ನಿಯಮಗಳ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ.",
            "mr": "रद्द करण्याचे शुल्क ऑर्डरची स्थिती आणि लागू खरेदी अटींवर अवलंबून असते.",
            "ta": "ரத்து கட்டணம் ஆர்டர் நிலை மற்றும் பொருந்தும் கொள்முதல் நிபந்தனைகளைப் பொறுத்தது.",
            "te": "రద్దు రుసుము ఆర్డర్ స్థితి మరియు వర్తించే కొనుగోలు నిబంధనలపై ఆధారపడి ఉంటుంది.",
        },

        "complaint": {
            "en": "I'm sorry that you had a problem. Please describe your complaint and I will help you with the next step.",
            "hi": "आपको हुई समस्या के लिए खेद है। कृपया अपनी शिकायत बताएं और मैं अगले चरण में आपकी मदद करूंगा।",
            "kn": "ನಿಮಗೆ ಸಮಸ್ಯೆ ಉಂಟಾದದ್ದಕ್ಕೆ ಕ್ಷಮಿಸಿ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ದೂರನ್ನು ವಿವರಿಸಿ, ಮುಂದಿನ ಹಂತದಲ್ಲಿ ನಾನು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ.",
            "mr": "तुम्हाला समस्या आली याबद्दल क्षमस्व. कृपया तुमची तक्रार सांगा आणि पुढील टप्प्यात मी मदत करतो.",
            "ta": "உங்களுக்கு ஏற்பட்ட சிக்கலுக்கு மன்னிக்கவும். உங்கள் புகாரைச் சொல்லுங்கள்; அடுத்த படியில் உதவுகிறேன்.",
            "te": "మీకు సమస్య ఎదురైనందుకు క్షమించండి. మీ ఫిర్యాదును వివరించండి, తదుపరి దశలో సహాయం చేస్తాను.",
        },

        "contact_customer_service": {
            "en": "Customer support is available to help you with your request.",
            "hi": "ग्राहक सहायता आपकी समस्या में मदद करने के लिए उपलब्ध है।",
            "kn": "ನಿಮ್ಮ ವಿನಂತಿಯಲ್ಲಿ ಸಹಾಯ ಮಾಡಲು ಗ್ರಾಹಕ ಬೆಂಬಲ ಲಭ್ಯವಿದೆ.",
            "mr": "तुमच्या विनंतीसाठी ग्राहक सहाय्य उपलब्ध आहे.",
            "ta": "உங்கள் கோரிக்கைக்கு உதவ வாடிக்கையாளர் ஆதரவு உள்ளது.",
            "te": "మీ అభ్యర్థనకు సహాయం చేయడానికి కస్టమర్ సపోర్ట్ అందుబాటులో ఉంది.",
        },

        "contact_human_agent": {
            "en": "I will connect you with a human support agent.",
            "hi": "मैं आपको एक मानव सहायता प्रतिनिधि से जोड़ता हूं।",
            "kn": "ನಿಮ್ಮನ್ನು ಮಾನವ ಸಹಾಯ ಪ್ರತಿನಿಧಿಯೊಂದಿಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
            "mr": "मी तुम्हाला मानव सहाय्य प्रतिनिधीशी जोडतो.",
            "ta": "உங்களை ஒரு மனித ஆதரவு பிரதிநிதியுடன் இணைக்கிறேன்.",
            "te": "మిమ్మల్ని మానవ సహాయ ప్రతినిధితో కలుపుతాను.",
        },

        "newsletter_subscription": {
            "en": "You can manage your newsletter subscription from your notification preferences.",
            "hi": "आप अपनी नोटिफिकेशन प्राथमिकताओं से न्यूज़लेटर सदस्यता प्रबंधित कर सकते हैं।",
            "kn": "ನಿಮ್ಮ ಅಧಿಸೂಚನೆ ಆದ್ಯತೆಗಳಿಂದ ಸುದ್ದಿಪತ್ರ ಚಂದಾದಾರಿಕೆಯನ್ನು ನಿರ್ವಹಿಸಬಹುದು.",
            "mr": "तुमच्या नोटिफिकेशन प्राधान्यांमधून वृत्तपत्र सदस्यता व्यवस्थापित करू शकता.",
            "ta": "உங்கள் அறிவிப்பு விருப்பங்களில் இருந்து நியூஸ்லெட்டர் சந்தாவை நிர்வகிக்கலாம்.",
            "te": "మీ నోటిఫికేషన్ ప్రాధాన్యతల నుండి వార్తాలేఖ సభ్యత్వాన్ని నిర్వహించవచ్చు.",
        },

        "registration_problems": {
            "en": "If registration is failing, check that your email is valid and all required fields are completed.",
            "hi": "पंजीकरण विफल होने पर जांचें कि आपका ईमेल सही है और सभी आवश्यक फ़ील्ड भरे गए हैं।",
            "kn": "ನೋಂದಣಿ ವಿಫಲವಾದರೆ ನಿಮ್ಮ ಇಮೇಲ್ ಸರಿಯಾಗಿದೆಯೇ ಮತ್ತು ಅಗತ್ಯವಿರುವ ಎಲ್ಲಾ ಕ್ಷೇತ್ರಗಳನ್ನು ಭರ್ತಿ ಮಾಡಿದ್ದೀರಾ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
            "mr": "नोंदणी अयशस्वी होत असल्यास तुमचा ईमेल योग्य आहे आणि सर्व आवश्यक माहिती भरली आहे का ते तपासा.",
            "ta": "பதிவு தோல்வியடைந்தால் உங்கள் மின்னஞ்சல் சரியானதா மற்றும் தேவையான அனைத்து புலங்களும் நிரப்பப்பட்டுள்ளனவா என்பதைச் சரிபார்க்கவும்.",
            "te": "నమోదు విఫలమైతే మీ ఇమెయిల్ సరైనదా మరియు అవసరమైన అన్ని ఫీల్డ్‌లు పూర్తి చేశారా అని తనిఖీ చేయండి.",
        },

        "review": {
            "en": "We would be happy to receive your feedback. Please share your review.",
            "hi": "हमें आपका फीडबैक पाकर खुशी होगी। कृपया अपनी समीक्षा साझा करें।",
            "kn": "ನಿಮ್ಮ ಪ್ರತಿಕ್ರಿಯೆಯನ್ನು ಸ್ವೀಕರಿಸಲು ನಮಗೆ ಸಂತೋಷವಾಗುತ್ತದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ವಿಮರ್ಶೆಯನ್ನು ಹಂಚಿಕೊಳ್ಳಿ.",
            "mr": "तुमचा अभिप्राय मिळाल्याचा आम्हाला आनंद होईल. कृपया तुमचे पुनरावलोकन सांगा.",
            "ta": "உங்கள் கருத்தைப் பெறுவதில் மகிழ்ச்சி. தயவுசெய்து உங்கள் மதிப்புரையைப் பகிரவும்.",
            "te": "మీ అభిప్రాయాన్ని స్వీకరించడానికి మేము సంతోషిస్తాము. దయచేసి మీ సమీక్షను పంచుకోండి.",
        },
    }

    def generate(
        self,
        router_result,
        workflow_result=None,
        language="en"
    ):

        if language not in self.LANGUAGES:
            language = "en"

        action = router_result.get("action", "handle_intent")
        intent = router_result.get("intent", "unknown")
        if action == "human_escalation":
            messages = {
                "en": "I'm not confident that I understood your request. I'll connect you with a support representative.",
                "hi": "मुझे आपकी बात पूरी तरह समझ नहीं आई। मैं आपको एक सहायता प्रतिनिधि से जोड़ता हूँ।",
                "kn": "ನಿಮ್ಮ ವಿನಂತಿಯನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ನಾನು ನಿಮ್ಮನ್ನು ಸಹಾಯ ಪ್ರತಿನಿಧಿಯೊಂದಿಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
                "mr": "मला तुमची विनंती पूर्णपणे समजली नाही. मी तुम्हाला सहाय्य प्रतिनिधीशी जोडतो.",
                "ta": "உங்கள் கோரிக்கையை முழுமையாக புரிந்துகொள்ள முடியவில்லை. உங்களை ஆதரவு பிரதிநிதியுடன் இணைக்கிறேன்.",
                "te": "మీ అభ్యర్థనను పూర్తిగా అర్థం చేసుకోలేకపోయాను. మిమ్మల్ని సహాయ ప్రతినిధితో కలుపుతాను.",
            }

            return messages[language]
        
        if action == "clarify":
            messages = {
                "en": "I want to make sure I understood you correctly. Could you please provide a little more detail?",
                "hi": "मैं यह सुनिश्चित करना चाहता हूं कि मैंने आपकी बात सही समझी है। कृपया थोड़ा और विवरण दें।",
                "kn": "ನಿಮ್ಮ ವಿನಂತಿಯನ್ನು ಸರಿಯಾಗಿ ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು ಬಯಸುತ್ತೇನೆ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ.",
                "mr": "मला तुमची विनंती योग्य प्रकारे समजली आहे याची खात्री करायची आहे. कृपया थोडी अधिक माहिती द्या.",
                "ta": "உங்கள் கோரிக்கையை நான் சரியாக புரிந்துகொண்டுள்ளேனா என்பதை உறுதிப்படுத்த விரும்புகிறேன். தயவுசெய்து மேலும் சில விவரங்களை வழங்கவும்.",
                "te": "మీ అభ్యర్థనను నేను సరిగ్గా అర్థం చేసుకున్నానని నిర్ధారிக்க விரும்பుతున్నాను. దయచేసి మరికొంత వివరాలు ఇవ్వండి.",
            }

            return messages[language]
        if not workflow_result:
            return get_response(
                language,
                "order_number"
            )

        status = workflow_result.get("status")
        if status == "need_information":
            if intent == "track_refund":
                return get_response(language, "refund_id")
            return get_response(language, "order_number")
        if status == "not_found":
            if intent == "track_refund":
                return get_response(language, "unknown_refund")
            return get_response(language, "unknown_order")
        if status == "needs_confirmation":
            return self._confirmation(intent, workflow_result, language)
        if status == "success" and intent in {
            "cancel_order",
            "get_refund",
            "delete_account",
            "change_order",
            "change_shipping_address",
        }:
            return self._confirmation_success(intent,workflow_result,language)
        data = workflow_result.get("data",{})
        if data:
            if intent == "track_order":
                return self._track_order(data,language)
            if intent == "track_refund":
                return self._track_refund(data,language)
        if intent in self.STATIC:
            return self.STATIC[intent].get(language,self.STATIC[intent]["en"])
        return workflow_result.get("message", "How can I help you today?")

    def _confirmation(
        self,
        intent,
        workflow_result,
        language
    ):
        data = workflow_result.get("data", {})
        order_id = data.get("order_id", "")
        messages = {
            "cancel_order": {
                "en": f"Order {order_id} can be cancelled. Do you want to confirm the cancellation?",
                "hi": f"ऑर्डर {order_id} को रद्द किया जा सकता है। क्या आप रद्द करने की पुष्टि करते हैं?",
                "kn": f"ಆರ್ಡರ್ {order_id} ಅನ್ನು ರದ್ದು ಮಾಡಬಹುದು. ರದ್ದತಿಯನ್ನು ಖಚಿತಪಡಿಸುತ್ತೀರಾ?",
                "mr": f"ऑर्डर {order_id} रद्द करता येतो. तुम्ही रद्द करण्याची पुष्टी करता का?",
                "ta": f"ஆர்டர் {order_id} ரத்து செய்யலாம். ரத்து செய்வதை உறுதிப்படுத்துகிறீர்களா?",
                "te": f"ఆర్డర్ {order_id} ను రద్దు చేయవచ్చు. రద్దును నిర్ధారించాలనుకుంటున్నారా?",
            },

            "get_refund": {
                "en": f"You can request a refund for order {order_id}. Do you want to proceed?",
                "hi": f"ऑर्डर {order_id} के लिए रिफंड का अनुरोध किया जा सकता है। क्या आप आगे बढ़ना चाहते हैं?",
                "kn": f"ಆರ್ಡರ್ {order_id} ಗೆ ರಿಫಂಡ್ ವಿನಂತಿಸಬಹುದು. ಮುಂದುವರಿಯಲು ಬಯಸುತ್ತೀರಾ?",
                "mr": f"ऑर्डर {order_id} साठी रिफंडची विनंती करता येईल. पुढे जायचे का?",
                "ta": f"ஆர்டர் {order_id} க்கு ரீஃபண்ட் கோரலாம். தொடர விரும்புகிறீர்களா?",
                "te": f"ఆర్డర్ {order_id} కు రీఫండ్ అభ్యర్థించవచ్చు. కొనసాగాలనుకుంటున్నారా?",
            },

            "delete_account": {
                "en": "Account deletion is permanent. Do you want to confirm?",
                "hi": "अकाउंट हटाना स्थायी है। क्या आप इसकी पुष्टि करते हैं?",
                "kn": "ಖಾತೆ ಅಳಿಸುವುದು ಶಾಶ್ವತವಾಗಿದೆ. ನೀವು ಖಚಿತಪಡಿಸುತ್ತೀರಾ?",
                "mr": "खाते हटवणे कायमस्वरूपी आहे. तुम्ही पुष्टी करता का?",
                "ta": "கணக்கை நீக்குவது நிரந்தரமானது. உறுதிப்படுத்த விரும்புகிறீர்களா?",
                "te": "ఖాతాను తొలగించడం శాశ్వతమైనది. మీరు నిర్ధారించాలనుకుంటున్నారా?",
            },

            "change_order": {
                "en": f"Order {order_id} can be modified. Do you want to continue?",
                "hi": f"ऑर्डर {order_id} में बदलाव किया जा सकता है। क्या आप जारी रखना चाहते हैं?",
                "kn": f"ಆರ್ಡರ್ {order_id} ಅನ್ನು ಬದಲಾಯಿಸಬಹುದು. ಮುಂದುವರಿಯಲು ಬಯಸುತ್ತೀರಾ?",
                "mr": f"ऑर्डर {order_id} मध्ये बदल करता येतो. पुढे जायचे का?",
                "ta": f"ஆர்டர் {order_id} ஐ மாற்றலாம். தொடர விரும்புகிறீர்களா?",
                "te": f"ఆర్డర్ {order_id} ను మార్చవచ్చు. కొనసాగాలనుకుంటున్నారా?",
            },
        }

        return messages.get(
            intent,
            messages["cancel_order"]
        ).get(
            language,
            messages.get(
                intent,
                messages["cancel_order"]
            )["en"]
        )

    def _confirmation_success(
        self,
        intent,
        workflow_result,
        language
    ):
        data = workflow_result.get("data", {})
        order_id = data.get("order_id", "")
        refund_id = data.get("refund_id", "")
        messages = {
            "cancel_order": {
                "en": f"Order {order_id} cancellation has been confirmed.",
                "hi": f"ऑर्डर {order_id} को रद्द करने की पुष्टि हो गई है।",
                "kn": f"ಆರ್ಡರ್ {order_id} ರದ್ದುಗೊಳಿಸಿರುವುದನ್ನು ಖಚಿತಪಡಿಸಲಾಗಿದೆ.",
                "mr": f"ऑर्डर {order_id} रद्द केल्याची पुष्टी झाली आहे.",
                "ta": f"ஆர்டர் {order_id} ரத்து செய்யப்பட்டதை உறுதிப்படுத்தியுள்ளோம்.",
                "te": f"ఆర్డర్ {order_id} రద్దు చేసినట్లు నిర్ధారించబడింది.",
            },

            "get_refund": {
                "en": (
                    f"Your refund request for order {order_id} "
                    f"has been confirmed. Refund ID: {refund_id}."
                ),
                "hi": (
                    f"ऑर्डर {order_id} के लिए आपके रिफंड अनुरोध की "
                    f"पुष्टि हो गई है। रिफंड आईडी: {refund_id}।"
                ),
                "kn": (
                    f"ಆರ್ಡರ್ {order_id} ಗಾಗಿ ನಿಮ್ಮ ರಿಫಂಡ್ ವಿನಂತಿಯನ್ನು "
                    f"ಖಚಿತಪಡಿಸಲಾಗಿದೆ. ರಿಫಂಡ್ ಐಡಿ: {refund_id}."
                ),
                "mr": (
                    f"ऑर्डर {order_id} साठी तुमच्या रिफंड विनंतीची "
                    f"पुष्टी झाली आहे. रिफंड आयडी: {refund_id}."
                ),
                "ta": (
                    f"ஆர்டர் {order_id} க்கான உங்கள் ரீஃபண்ட் கோரிக்கை "
                    f"உறுதிப்படுத்தப்பட்டது. ரீஃபண்ட் ஐடி: {refund_id}."
                ),
                "te": (
                    f"ఆర్డర్ {order_id} కోసం మీ రీఫండ్ అభ్యర్థన "
                    f"నిర్ధారించబడింది. రీఫండ్ ID: {refund_id}."
                ),
            },

            "delete_account": {
                "en": "Your account deletion request has been confirmed.",
                "hi": "आपके अकाउंट हटाने के अनुरोध की पुष्टि हो गई है।",
                "kn": "ನಿಮ್ಮ ಖಾತೆ ಅಳಿಸುವ ವಿನಂತಿಯನ್ನು ಖಚಿತಪಡಿಸಲಾಗಿದೆ.",
                "mr": "तुमच्या खाते हटवण्याच्या विनंतीची पुष्टी झाली आहे.",
                "ta": "உங்கள் கணக்கு நீக்கக் கோரிக்கை உறுதிப்படுத்தப்பட்டுள்ளது.",
                "te": "మీ ఖాతా తొలగింపు అభ్యర్థన నిర్ధారించబడింది.",
            },

            "change_order": {
                "en": (
                    f"Order {order_id} modification has been confirmed. "
                    "Please provide the changes you want to make."
                ),
                "hi": (
                    f"ऑर्डर {order_id} में बदलाव की पुष्टि हो गई है। "
                    "कृपया बताएं कि आप क्या बदलाव करना चाहते हैं।"
                ),
                "kn": (
                    f"ಆರ್ಡರ್ {order_id} ಬದಲಾವಣೆಯನ್ನು ಖಚಿತಪಡಿಸಲಾಗಿದೆ. "
                    "ನೀವು ಮಾಡಲು ಬಯಸುವ ಬದಲಾವಣೆಗಳನ್ನು ತಿಳಿಸಿ."
                ),
                "mr": (
                    f"ऑर्डर {order_id} मधील बदलाची पुष्टी झाली आहे. "
                    "तुम्हाला कोणते बदल करायचे आहेत ते सांगा."
                ),
                "ta": (
                    f"ஆர்டர் {order_id} மாற்றம் உறுதிப்படுத்தப்பட்டுள்ளது. "
                    "நீங்கள் செய்ய விரும்பும் மாற்றங்களைத் தெரிவிக்கவும்."
                ),
                "te": (
                    f"ఆర్డర్ {order_id} మార్పు నిర్ధారించబడింది. "
                    "మీరు చేయాలనుకుంటున్న మార్పులను చెప్పండి."
                ),
            },

            "change_shipping_address": {
                "en": (
                    f"Shipping address change for order {order_id} "
                    "has been confirmed. Please provide the new address."
                ),
                "hi": (
                    f"ऑर्डर {order_id} के शिपिंग पते में बदलाव की पुष्टि हो गई है। "
                    "कृपया नया पता बताएं।"
                ),
                "kn": (
                    f"ಆರ್ಡರ್ {order_id} ರ ಶಿಪ್ಪಿಂಗ್ ವಿಳಾಸ ಬದಲಾವಣೆಯನ್ನು "
                    "ಖಚಿತಪಡಿಸಲಾಗಿದೆ. ದಯವಿಟ್ಟು ಹೊಸ ವಿಳಾಸವನ್ನು ನೀಡಿ."
                ),
                "mr": (
                    f"ऑर्डर {order_id} च्या शिपिंग पत्त्यातील बदलाची पुष्टी झाली आहे. "
                    "कृपया नवीन पत्ता द्या."
                ),
                "ta": (
                    f"ஆர்டர் {order_id} இன் ஷிப்பிங் முகவரி மாற்றம் "
                    "உறுதிப்படுத்தப்பட்டுள்ளது. புதிய முகவரியை வழங்கவும்."
                ),
                "te": (
                    f"ఆర్డర్ {order_id} షిప్పింగ్ చిరునామా మార్పు "
                    "నిర్ధారించబడింది. దయచేసి కొత్త చిరునామాను ఇవ్వండి."
                ),
            },
        }

        language_messages = messages.get(intent, messages["cancel_order"])
        selected_language = language if language in self.LANGUAGES else "en"
        return language_messages.get(
            selected_language,
            language_messages["en"]
        )


    def _track_order(self, data, language):
        order_id = data.get("order_id", "")
        status = data.get("status", "")
        item = data.get("item", "")
        delivery_date = data.get("delivery_date", "")
        status_map = {
            "en": {
                "Shipped": "Shipped",
                "Processing": "Processing",
                "Delivered": "Delivered",
            },
            "hi": {
                "Shipped": "भेज दिया गया",
                "Processing": "प्रक्रिया में है",
                "Delivered": "डिलीवर हो गया",
            },
            "kn": {
                "Shipped": "ರವಾನೆಯಾಗಿದೆ",
                "Processing": "ಪ್ರಕ್ರಿಯೆಯಲ್ಲಿದೆ",
                "Delivered": "ತಲುಪಿಸಲಾಗಿದೆ",
            },
            "mr": {
                "Shipped": "पाठवण्यात आला आहे",
                "Processing": "प्रक्रियेत आहे",
                "Delivered": "वितरित करण्यात आला आहे",
            },
            "ta": {
                "Shipped": "அனுப்பப்பட்டுள்ளது",
                "Processing": "செயல்பாட்டில் உள்ளது",
                "Delivered": "விநியோகிக்கப்பட்டது",
            },
            "te": {
                "Shipped": "పంపబడింది",
                "Processing": "ప్రక్రియలో ఉంది",
                "Delivered": "డెలివరీ చేయబడింది",
            },
        }

        localized_status = status_map.get(
            language,
            status_map["en"]
        ).get(status, status)
        templates = {
            "en": (
                f"Your order {order_id} is currently "
                f"{localized_status}. "
                f"Your {item} is expected to arrive by "
                f"{delivery_date}."
            ),

            "hi": (
                f"आपका ऑर्डर {order_id} अभी "
                f"{localized_status} है। "
                f"आपका {item} {delivery_date} तक "
                f"पहुंचने की उम्मीद है।"
            ),

            "kn": (
                f"ನಿಮ್ಮ ಆರ್ಡರ್ {order_id} ಪ್ರಸ್ತುತ "
                f"{localized_status}. "
                f"ನಿಮ್ಮ {item} {delivery_date} ರಂದು "
                f"ತಲುಪುವ ನಿರೀಕ್ಷೆಯಿದೆ."
            ),

            "mr": (
                f"तुमचा ऑर्डर {order_id} सध्या "
                f"{localized_status}. "
                f"तुमचा {item} {delivery_date} पर्यंत "
                f"पोहोचण्याची अपेक्षा आहे."
            ),

            "ta": (
                f"உங்கள் ஆர்டர் {order_id} தற்போது "
                f"{localized_status}. "
                f"உங்கள் {item} {delivery_date} அன்று "
                f"வந்து சேரும் என எதிர்பார்க்கப்படுகிறது."
            ),

            "te": (
                f"మీ ఆర్డర్ {order_id} ప్రస్తుతం "
                f"{localized_status}. "
                f"మీ {item} {delivery_date} నాటికి "
                f"చేరుతుందని భావిస్తున్నారు."
            ),
        }

        return templates.get(
            language,
            templates["en"]
        )

    def _track_refund(
        self,
        data,
        language
    ):

        refund_id = data.get("refund_id", "")
        status = data.get("status","")
        amount = data.get("amount",0)
        status_map = {

            "en": {
                "Processing": "Processing",
                "Completed": "Completed",
            },

            "hi": {
                "Processing": "प्रक्रिया में है",
                "Completed": "पूरा हो गया है",
            },

            "kn": {
                "Processing": "ಪ್ರಕ್ರಿಯೆಯಲ್ಲಿದೆ",
                "Completed": "ಪೂರ್ಣಗೊಂಡಿದೆ",
            },

            "mr": {
                "Processing": "प्रक्रियेत आहे",
                "Completed": "पूर्ण झाले आहे",
            },

            "ta": {
                "Processing": "செயல்பாட்டில் உள்ளது",
                "Completed": "முடிந்துவிட்டது",
            },

            "te": {
                "Processing": "ప్రక్రియలో ఉంది",
                "Completed": "పూర్తయింది",
            },
        }

        localized_status = status_map.get(
            language,
            status_map["en"]
        ).get(status, status)
        templates = {

            "en": (
                f"Refund {refund_id} is currently "
                f"{localized_status}. "
                f"The refund amount is ₹{amount}."
            ),

            "hi": (
                f"रिफंड {refund_id} अभी "
                f"{localized_status}। "
                f"रिफंड राशि ₹{amount} है।"
            ),

            "kn": (
                f"ರಿಫಂಡ್ {refund_id} ಪ್ರಸ್ತುತ "
                f"{localized_status}. "
                f"ರಿಫಂಡ್ ಮೊತ್ತ ₹{amount}."
            ),

            "mr": (
                f"रिफंड {refund_id} सध्या "
                f"{localized_status}. "
                f"रिफंडची रक्कम ₹{amount} आहे."
            ),

            "ta": (
                f"ரீஃபண்ட் {refund_id} தற்போது "
                f"{localized_status}. "
                f"ரீஃபண்ட் தொகை ₹{amount}."
            ),

            "te": (
                f"రీఫండ్ {refund_id} ప్రస్తుతం "
                f"{localized_status}. "
                f"రీఫండ్ మొత్తం ₹{amount}."
            ),
        }
        return templates.get(language, templates["en"])
