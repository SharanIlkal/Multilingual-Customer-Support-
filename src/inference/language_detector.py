import re


class LanguageDetector:

    def detect(self, text):

        if not text:
            return "en"

        # Kannada
        if re.search(
            r"[\u0C80-\u0CFF]",
            text
        ):
            return "kn"

        # Telugu
        if re.search(
            r"[\u0C00-\u0C7F]",
            text
        ):
            return "te"

        # Tamil
        if re.search(
            r"[\u0B80-\u0BFF]",
            text
        ):
            return "ta"

        # Devanagari
        if re.search(
            r"[\u0900-\u097F]",
            text
        ):

            marathi_words = [
                "मला",
                "माझा",
                "माझी",
                "माझे",
                "माझ्या",
                "तुमचा",
                "तुमची",
                "तुमचे",
                "तुमच्या",
                "आहे",
                "आहेत",
                "करायचे",
                "करायची",
                "करायचा",
                "पाहिजे",
                "हवे",
                "हवी",
                "मिळेल",
                "सांगा",
                "घ्यायचा",
                "रद्द करायची",
                "ऑर्डरमध्ये",
                "रिफंडची",
                "डिलिव्हरीला",
                "ट्रॅक",
            ]

            marathi_score = sum(
                word in text
                for word in marathi_words
            )

            if marathi_score > 0:
                return "mr"

            return "hi"

        return "en"