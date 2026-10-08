from .conversation_state import ConversationState
from .intent_router import IntentRouter
from .workflow_handler import WorkflowHandler
from .response_generator import ResponseGenerator
from .entity_extractor import EntityExtractor
from .language_detector import LanguageDetector
import re

class SupportChatbot:

    def __init__(self, model, vocab, id_to_label):
        self.state = ConversationState()
        self.router = IntentRouter(model, vocab, id_to_label)
        self.workflow = WorkflowHandler()
        self.response_generator = ResponseGenerator()
        self.entity_extractor = EntityExtractor()
        self.language_detector = LanguageDetector()

    def process_message(self, message):
        message = message.strip()
        if not message:
            return self._result(
                "Please enter a message.",
                None,
                None,
                0.0,
                "clarify"
            )

        self.state.add_message("user", message)
        detected_language = self.language_detector.detect(message)
        previous_language = self.state.get_context("language")
        if (
            previous_language
            and (self.state.is_awaiting_confirmation() or self.state.is_awaiting_entity() or self._is_short_followup(message))
        ):
            language = previous_language
        else:
            language = detected_language

        self.state.set_context("language", language)

        if self.state.is_awaiting_entity():
            expected_entity = (self.state.get_expected_entity())
            entities = self.entity_extractor.extract(message, expected_entity=expected_entity)
            if expected_entity == "order_id":
                if "order_id" in entities:
                    order_id = entities["order_id"]
                    self.state.set_context("order_id", order_id)
                    self.state.clear_expected_entity()
                    return self._execute_current_intent(language)
                return self._entity_error("order_id", language)
            if expected_entity == "refund_id":
                if "refund_id" in entities:
                    refund_id = entities["refund_id"]
                    self.state.set_context("refund_id", refund_id)
                    self.state.clear_expected_entity()
                    return self._execute_current_intent(language)
                return self._entity_error("refund_id", language)

        if self.state.is_awaiting_confirmation():
            confirmation = self._parse_confirmation(message)
            if confirmation is True:
                intent = self.state.current_intent
                category = self.state.current_category
                result = self.workflow.confirm(intent, self.state.context)
                self.state.clear_confirmation()
                if intent in {
                    "track_order",
                    "get_refund",
                    "cancel_order",
                    "change_order",
                    "change_shipping_address",
                    "check_invoice",
                    "get_invoice",
                }:
                    self.state.set_context(
                        "order_id",
                        None
                    )

                if intent == "track_refund":
                    self.state.set_context(
                        "refund_id",
                        None
                    )

                response = self.response_generator.generate(
                    {
                        "action": "handle_intent",
                        "intent": intent,
                        "category": category,
                        "confidence": 1.0
                    },
                    result,
                    language=language
                )

                return self._result(
                    response,
                    intent,
                    category,
                    1.0,
                    "handle_intent",
                    result
                )

            if confirmation is False:

                intent = self.state.current_intent
                category = self.state.current_category

                self.state.clear_confirmation()

                # User cancelled the confirmation, so the
                # pending entity is no longer needed.
                if intent in {
                    "track_order",
                    "get_refund",
                    "cancel_order",
                    "change_order",
                    "change_shipping_address",
                    "check_invoice",
                    "get_invoice",
                }:
                    self.state.set_context(
                        "order_id",
                        None
                    )

                if intent == "track_refund":
                    self.state.set_context(
                        "refund_id",
                        None
                    )

                response = self._localized(
                    language,
                    {
                        "en": "Okay. I cancelled that request.",
                        "hi": "ठीक है। मैंने उस अनुरोध को रद्द कर दिया है।",
                        "kn": "ಸರಿ. ಆ ವಿನಂತಿಯನ್ನು ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ.",
                        "mr": "ठीक आहे. मी ती विनंती रद्द केली आहे.",
                        "ta": "சரி. அந்த கோரிக்கையை நான் ரத்து செய்துவிட்டேன்.",
                        "te": "సరే. ఆ అభ్యర్థనను నేను రద్దు చేశాను.",
                    }
                )

                return self._result(
                    response,
                    intent,
                    category,
                    1.0,
                    "handle_intent"
                )

            response = self._localized(
                language,
                {
                    "en": "Please reply yes to confirm or no to cancel.",
                    "hi": "पुष्टि करने के लिए हाँ या रद्द करने के लिए नहीं कहें।",
                    "kn": "ದೃಢೀಕರಿಸಲು ಹೌದು ಅಥವಾ ರದ್ದುಗೊಳಿಸಲು ಇಲ್ಲ ಎಂದು ಉತ್ತರಿಸಿ.",
                    "mr": "पुष्टी करण्यासाठी होय किंवा रद्द करण्यासाठी नाही असे उत्तर द्या.",
                    "ta": "உறுதிப்படுத்த ஆம் அல்லது ரத்து செய்ய இல்லை என்று பதிலளிக்கவும்.",
                    "te": "నిర్ధారించడానికి అవును లేదా రద్దు చేయడానికి కాదు అని సమాధానం ఇవ్వండి.",
                }
            )

            return self._result(
                response,
                self.state.current_intent,
                self.state.current_category,
                1.0,
                "clarify"
            )

        # =====================================================
        # 3. ENTITY EXTRACTION
        # =====================================================

        entities = self.entity_extractor.extract(
            message
        )

        # Persist only genuinely extracted entity values.
        # The extractor supports arbitrary IDs, but normal words
        # such as "order" must never be stored as an ID.
        for key, value in entities.items():
            if value is not None and str(value).strip():
                self.state.set_context(
                    key,
                    str(value).strip()
                )

        # =====================================================
        # 4. PREVIOUS INTENT
        # =====================================================

        previous_intent = self.state.current_intent
        previous_category = self.state.current_category

        # =====================================================
        # 5. CLARIFICATION FOLLOW-UP
        # =====================================================

        if self.state.is_awaiting_clarification():

            matched_intent = self._resolve_clarification(
                message
            )

            if matched_intent:

                # A clarification answer selects a new intent.
                # Clear any stale entity from the previous intent
                # before executing the selected intent.
                if matched_intent != previous_intent:
                    self.state.set_context(
                        "order_id",
                        None
                    )
                    self.state.set_context(
                        "refund_id",
                        None
                    )

                self.state.set_intent(
                    matched_intent,
                    previous_category
                )

                self.state.set_clarification(
                    False
                )

                return self._execute_current_intent(
                    language
                )

        # =====================================================
        # 6. ID-ONLY FOLLOW-UP
        # =====================================================

        if self._is_id_only(message):

            if previous_intent:

                if previous_intent in {
                    "track_order",
                    "get_refund",
                    "cancel_order",
                    "change_order",
                    "change_shipping_address",
                    "check_invoice",
                    "get_invoice",
                }:

                    order_id = (
                        self.entity_extractor.extract_order_id(
                            message
                        )
                    )

                    if order_id:

                        self.state.set_context(
                            "order_id",
                            order_id
                        )

                        self.state.clear_expected_entity()

                        return self._execute_current_intent(
                            language
                        )

                if previous_intent == "track_refund":

                    refund_id = (
                        self.entity_extractor.extract_refund_id(
                            message
                        )
                    )

                    if refund_id:

                        self.state.set_context(
                            "refund_id",
                            refund_id
                        )

                        self.state.clear_expected_entity()

                        return self._execute_current_intent(
                            language
                        )

        # =====================================================
        # 7. NORMAL INTENT CLASSIFICATION
        # =====================================================

        router_result = self.router.classify(
            message
        )

        # Explicit Kannada cancellation guard.
        normalized_message = re.sub(
            r"\s+",
            " ",
            message.lower().strip()
        )

        kannada_cancel_phrases = {
            "ನನ್ನ ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
            "ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
            "ನನ್ನ ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
            "ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
        }

        if any(
            phrase in normalized_message
            for phrase in kannada_cancel_phrases
        ):
            router_result["intent"] = "cancel_order"
            router_result["action"] = "handle_intent"
            router_result["confidence"] = max(
                float(router_result.get("confidence", 0.0)),
                0.99
            )

        # =====================================================
        # 8. MULTILINGUAL OVERRIDES
        # =====================================================

        router_result = self._apply_intent_overrides(
            message,
            router_result
        )

        # Final Kannada cancellation guard.
        # This runs after all generic overrides, so broad phrases such as
        # "ನನ್ನ ಆರ್ಡರ್" can never overwrite an explicit cancellation request.
        normalized_message = re.sub(
            r"\s+",
            " ",
            message.lower().strip()
        )

        kannada_cancel_phrases = {
            "ನನ್ನ ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
            "ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
            "ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
            "ನನ್ನ ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
            "ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
        }

        if any(
            phrase in normalized_message
            for phrase in kannada_cancel_phrases
        ):
            router_result["intent"] = "cancel_order"
            router_result["action"] = "handle_intent"
            router_result["confidence"] = max(
                float(router_result.get("confidence", 0.0)),
                0.99
            )

        new_intent = router_result["intent"]

        # A new natural-language intent must not inherit an
        # entity belonging to a previous intent.
        #
        # We do this AFTER classification, so an ID follow-up
        # is never mistaken for a new request.
        if (
            previous_intent is not None
            and new_intent != previous_intent
        ):
            self.state.set_context(
                "order_id",
                None
            )
            self.state.set_context(
                "refund_id",
                None
            )

        self.state.clear_expected_entity()

        self.state.set_intent(
            new_intent,
            router_result["category"]
        )

        # =====================================================
        # 9. CLARIFICATION / ESCALATION
        # =====================================================

        if router_result["action"] != "handle_intent":

            if router_result["action"] == "clarify":

                self.state.set_clarification(
                    True
                )

            response = self.response_generator.generate(
                router_result,
                language=language
            )

            return self._result(
                response,
                router_result["intent"],
                router_result["category"],
                router_result["confidence"],
                router_result["action"]
            )

        # =====================================================
        # 10. WORKFLOW
        # =====================================================

        return self._execute_current_intent(
            language,
            router_result
        )

    # =========================================================
    # EXECUTE INTENT
    # =========================================================

    def _execute_current_intent(
        self,
        language,
        router_result=None
    ):
        intent = self.state.current_intent
        category = self.state.current_category

        # ---------------------------------------------------------
        # Required entities
        # ---------------------------------------------------------

        entity_requirements = {
            "track_order": "order_id",
            "get_refund": "order_id",
            "cancel_order": "order_id",
            "change_order": "order_id",
            "change_shipping_address": "order_id",
            "check_invoice": "order_id",
            "get_invoice": "order_id",
            "track_refund": "refund_id",
        }

        required_entity = entity_requirements.get(intent)

        # ---------------------------------------------------------
        # Check required entity
        # ---------------------------------------------------------

        if required_entity:
            value = self.state.get_context(required_entity)

            if not value:
                self.state.set_expected_entity(required_entity)

                if required_entity == "order_id":
                    messages = {
                        "en": "Sure. Please provide your order number.",
                        "hi": "ज़रूर। कृपया अपना ऑर्डर नंबर बताएं।",
                        "kn": "ಖಂಡಿತ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಆರ್ಡರ್ ಸಂಖ್ಯೆಯನ್ನು ನೀಡಿ.",
                        "mr": "नक्की. कृपया तुमचा ऑर्डर नंबर सांगा.",
                        "ta": "நிச்சயமாக. தயவுசெய்து உங்கள் ஆர்டர் எண்ணை வழங்கவும்.",
                        "te": "తప్పకుండా. దయచేసి మీ ఆర్డర్ నంబర్‌ను ఇవ్వండి.",
                    }
                else:
                    messages = {
                        "en": "Sure. Please provide your refund ID.",
                        "hi": "ज़रूर। कृपया अपना रिफंड आईडी बताएं।",
                        "kn": "ಖಂಡಿತ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ರಿಫಂಡ್ ಐಡಿಯನ್ನು ನೀಡಿ.",
                        "mr": "नक्की. कृपया तुमचा रिफंड आयडी सांगा.",
                        "ta": "நிச்சயமாக. தயவுசெய்து உங்கள் ரீஃபண்ட் ஐடியை வழங்கவும்.",
                        "te": "తప్పకుండా. దయచేసి మీ రీఫండ్ IDని ఇవ్వండి.",
                    }

                response = self._localized(
                    language,
                    messages
                )

                return self._result(
                    response,
                    intent,
                    category,
                    (
                        router_result["confidence"]
                        if router_result
                        else 1.0
                    ),
                    "handle_intent"
                )

        # ---------------------------------------------------------
        # Execute workflow
        # ---------------------------------------------------------

        workflow_result = self.workflow.handle(
            intent,
            self.state.context
        )

        # ---------------------------------------------------------
        # Confirmation-required workflows
        # ---------------------------------------------------------

        if workflow_result.get(
            "requires_confirmation",
            False
        ):
            self.state.set_confirmation(
                True,
                intent
            )

        # ---------------------------------------------------------
        # Generate response
        # ---------------------------------------------------------

        response = self.response_generator.generate(
            {
                "action": "handle_intent",
                "intent": intent,
                "category": category,
                "confidence": (
                    router_result["confidence"]
                    if router_result
                    else 1.0
                )
            },
            workflow_result,
            language=language
        )

        confidence = (
            router_result["confidence"]
            if router_result
            else 1.0
        )

        # ---------------------------------------------------------
        # Clear consumed entity ONLY after successful workflow
        #
        # Do NOT clear it when confirmation is pending.
        # The confirmation step still needs the context.
        # ---------------------------------------------------------

        if (
            required_entity
            and not workflow_result.get(
                "requires_confirmation",
                False
            )
        ):
            self.state.set_context(
                required_entity,
                None
            )

        return self._result(
            response,
            intent,
            category,
            confidence,
            "handle_intent",
            workflow_result
        )

    # =========================================================
    # ENTITY ERROR
    # =========================================================

    def _entity_error(
        self,
        entity,
        language
    ):

        if entity == "order_id":

            response = self._localized(
                language,
                {
                    "en": "Please provide a valid order ID.",
                    "hi": "कृपया एक मान्य ऑर्डर आईडी दें।",
                    "kn": "ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ ಆರ್ಡರ್ ಐಡಿಯನ್ನು ನೀಡಿ.",
                    "mr": "कृपया वैध ऑर्डर आयडी द्या.",
                    "ta": "தயவுசெய்து சரியான ஆர்டர் ஐடியை வழங்கவும்.",
                    "te": "దయచేసి సరైన ఆర్డర్ IDని ఇవ్వండి.",
                }
            )

        else:

            response = self._localized(
                language,
                {
                    "en": "Please provide a valid refund ID.",
                    "hi": "कृपया एक मान्य रिफंड आईडी दें।",
                    "kn": "ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ ರಿಫಂಡ್ ಐಡಿಯನ್ನು ನೀಡಿ.",
                    "mr": "कृपया वैध रिफंड आयडी द्या.",
                    "ta": "தயவுசெய்து சரியான ரீஃபண்ட் ஐடியை வழங்கவும்.",
                    "te": "దయచేసి సరైన రీఫండ్ IDని ఇవ్వండి.",
                }
            )

        return self._result(
            response,
            self.state.current_intent,
            self.state.current_category,
            1.0,
            "clarify"
        )

    # =========================================================
    # CONFIRMATION PARSER
    # =========================================================

    def _parse_confirmation(
        self,
        message
    ):

        text = message.strip().lower()

        yes_words = {
            "yes",
            "y",
            "yeah",
            "yep",
            "sure",
            "okay",
            "ok",
            "confirm",
            "confirmed",
            "proceed",

            "हाँ",
            "हां",
            "जी हाँ",

            "ಹೌದು",
            "ಸರಿ",

            "हो",
            "होय",

            "ஆம்",
            "சரி",

            "అవును",
            "సరే",
        }

        no_words = {
            "no",
            "n",
            "nope",
            "cancel",
            "stop",

            "नहीं",
            "नही",

            "ಇಲ್ಲ",

            "नाही",

            "இல்லை",

            "కాదు",
        }

        if text in yes_words:
            return True

        if text in no_words:
            return False

        return None

    # =========================================================
    # INTENT OVERRIDES
    # =========================================================

    def _apply_intent_overrides(
        self,
        message,
        router_result
    ):

        # IntentRouter is the single source of routing decisions.  Its
        # multilingual overrides are more specific than the legacy rules
        # below, whose broad order-tracking phrases can otherwise replace a
        # correctly matched request to change an order.
        if router_result.get("routing_source") != "model":
            return router_result

        text = message.lower()

        overrides = {

            "track_order": [
                "track my order",
                "track order",
                "order status",
                "status of my order",
                "मेरा ऑर्डर ट्रैक",
                "ऑर्डर ट्रैक",
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
                "मेरा रिफंड",
                "रिफंड ट्रैक",
                "रिफंड की स्थिति",
                "ನನ್ನ ರಿಫಂಡ್",
                "ರಿಫಂಡ್ ಟ್ರ್ಯಾಕ್",
                "ರಿಫಂಡ್ ಸ್ಥಿತಿ",
                "माझा रिफंड",
                "रिफंड ट्रॅक",
                "ரீஃபண்ட் நிலை",
                "எனது ரீஃபண்ட்",
                "రీఫండ్ స్థితి",
                "నా రీఫండ్",
            ],

            "get_refund": [
                "i want a refund",
                "want a refund",
                "request a refund",
                "get a refund",
                "refund for my order",
                "need a refund",
                "रिफंड चाहिए",
                "रिफंड लेना है",
                "मुझे रिफंड चाहिए",
                "ರಿಫಂಡ್ ಬೇಕು",
                "ರಿಫಂಡ್ ಪಡೆಯಬೇಕು",
                "मला रिफंड हवा",
                "ரீஃபண்ட் வேண்டும்",
                "ரீஃபண்ட் பெற வேண்டும்",
                "నాకు రీఫండ్ కావాలి",
                "రీఫండ్ కావాలి",
            ],

            "cancel_order": [
                "cancel my order",
                "cancel order",
                "want to cancel my order",
                "ऑर्डर रद्द",
                "ऑर्डर कैंसल",
                "आर्डर रद्द",
                "ಆರ್ಡರ್ ರದ್ದು",
                "ನನ್ನ ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
                "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
                "ಆರ್ಡರ್ ಅನ್ನು ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",
                "ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಬೇಕು",
                "ನನ್ನ ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
                "ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್ ಮಾಡಬೇಕು",
                "ಆರ್ಡರ್ ಕ್ಯಾನ್ಸಲ್",
                "ऑर्डर रद्द करायचा",
                "ऑर्डर कॅन्सल",
                "ஆர்டரை ரத்து",
                "ஆர்டர் கேன்சல்",
                "ఆర్డర్ రద్దు",
                "ఆర్డర్ క్యాన్సల్",
            
                "ನನ್ನ ಆರ್ಡರ್ ರದ್ದುಗೊಳಿಸಲು ಬಯಸುತ್ತೇನೆ",],
        }

        # Check specific intents before broad track_order phrases.
        # For example, "ನನ್ನ ಆರ್ಡರ್ ..." is a broad track_order phrase,
        # but explicit cancellation wording must remain cancel_order.
        override_order = [
            "cancel_order",
            "get_refund",
            "track_refund",
            "track_order",
        ]

        for intent in override_order:
            phrases = overrides[intent]

            if any(
                phrase in text
                for phrase in phrases
            ):

                router_result["intent"] = intent
                router_result["action"] = "handle_intent"
                router_result["confidence"] = max(
                    router_result["confidence"],
                    0.99
                )

                break

        return router_result

    # =========================================================
    # CLARIFICATION RESOLUTION
    # =========================================================

    def _resolve_clarification(
        self,
        message
    ):

        text = message.lower()

        mappings = {

            "delivery_period": [
                "how long",
                "how many days",
                "delivery time",
                "delivery period",
                "when will",
                "how long will delivery take",
                "कितने दिन",
                "कितना समय",
                "ಎಷ್ಟು ದಿನ",
                "ಎಷ್ಟು ಸಮಯ",
                "किती दिवस",
                "किती वेळ",
                "எத்தனை நாட்கள்",
                "ఎన్ని రోజులు",
            ],

            "delivery_options": [
                "delivery option",
                "delivery options",
                "shipping options",
                "express delivery",
                "standard delivery",
            ],

            "change_order": [
                "change my order",
                "modify my order",
                "edit my order",
                "change order",
                "modify order",
            ],

            "get_refund": [
                "want a refund",
                "get a refund",
                "request a refund",
                "refund for my order",
            ],

            "track_refund": [
                "track refund",
                "refund status",
                "refund tracking",
            ],
        }

        for intent, phrases in mappings.items():

            if any(
                phrase in text
                for phrase in phrases
            ):
                return intent

        return None

    # =========================================================
    # HELPERS
    # =========================================================

    def _is_id_only(self, message):

        text = message.strip()

        return bool(
            self.entity_extractor.extract_generic_id(
                text
            )
        )

    def _is_short_followup(self, message):

        text = message.strip().lower()

        return (
            len(text.split()) <= 3
            or self._is_id_only(message)
            or self._parse_confirmation(message) is not None
        )

    def _localized(
        self,
        language,
        messages
    ):

        return messages.get(
            language,
            messages["en"]
        )

    def _result(
        self,
        response,
        intent,
        category,
        confidence,
        action,
        workflow_result=None
    ):

        self.state.add_message(
            "assistant",
            response
        )

        result = {
            "response": response,
            "intent": intent,
            "category": category,
            "confidence": confidence,
            "action": action,
        }

        if workflow_result:

            result["workflow_status"] = (
                workflow_result.get("status")
            )

        return result

    # =========================================================
    # PUBLIC HELPERS
    # =========================================================

    def set_context(self, key, value):

        self.state.set_context(
            key,
            value
        )

    def get_history(self):

        return self.state.get_history()

    def reset(self):

        self.state.clear()
