import re

class EntityExtractor:

    def extract(self, message, intent=None, expected_entity=None):
        entities = {}
        if not message:
            return entities
        text = message.strip()
        if expected_entity == "order_id":
            order_id = self.extract_order_id(text)
            if order_id:
                entities["order_id"] = order_id
            return entities
        if expected_entity == "refund_id":
            refund_id = self.extract_refund_id(text)
            if refund_id:
                entities["refund_id"] = refund_id
            return entities
        order_id = self.extract_order_id(text)
        if order_id:
            entities["order_id"] = order_id
        refund_id = self.extract_refund_id(text)
        if refund_id:
            entities["refund_id"] = refund_id
        return entities

    def extract_order_id(self, text):
        text = text.strip()
        match = re.search(
            r"\b(?:order|ord)(?:[\s:#_-]+([A-Za-z0-9][A-Za-z0-9_-]*)|(?=[0-9])([0-9][A-Za-z0-9_-]*))\b",
            text,
            re.IGNORECASE
        )
        if match:
            return match.group(1) or match.group(2)
        if re.fullmatch(r"\d+", text):
            return text
        if re.fullmatch(
            r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*",
            text
        ):
            if any(char.isdigit() for char in text):
                return text
        return None

    def extract_refund_id(self, text):
        text = text.strip()
        if re.fullmatch(r"\d+", text):
            return text
        match = re.search(
            r"\b(?:refund|ref|refundid|refund_id)"
            r"[\s:#_-]*([A-Za-z0-9][A-Za-z0-9_-]*)\b",
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1)

        match = re.fullmatch(
            r"R[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*",
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0).upper()
        return None


    def extract_generic_id(self, text):
        text = text.strip()
        if re.fullmatch(r"\d+", text):
            return text
        if re.fullmatch(
            r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*",
            text
        ):
            if any(char.isdigit() for char in text):
                return text
        return None
