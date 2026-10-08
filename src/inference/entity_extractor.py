import re


class EntityExtractor:

    # =========================================================
    # MAIN EXTRACTION
    # =========================================================

    def extract(self, message, intent=None, expected_entity=None):

        entities = {}

        if not message:
            return entities

        text = message.strip()

        # -----------------------------------------------------
        # If chatbot explicitly expects an entity, prioritize it
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Normal extraction
        # -----------------------------------------------------

        order_id = self.extract_order_id(text)

        if order_id:
            entities["order_id"] = order_id

        refund_id = self.extract_refund_id(text)

        if refund_id:
            entities["refund_id"] = refund_id

        return entities

    # =========================================================
    # ORDER ID
    # =========================================================

    def extract_order_id(self, text):

        text = text.strip()

        # -----------------------------------------------------
        # Explicit order labels
        #
        # ORDER 12
        # ORDER-12
        # ORDER_12
        # ORD 584
        # ORD-584
        # -----------------------------------------------------

        # Explicit order labels.
        #
        # Valid:
        #   ORDER 12
        #   ORDER-584
        #   ORDER_1478
        #   ORD 584
        #   ORD123
        #   ORDER123
        #
        # Important: plain words such as "order" must NOT become
        # an ID such as "er".
        match = re.search(
            r"\b(?:order|ord)(?:[\s:#_-]+([A-Za-z0-9][A-Za-z0-9_-]*)|(?=[0-9])([0-9][A-Za-z0-9_-]*))\b",
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1) or match.group(2)

        # -----------------------------------------------------
        # Standalone numeric ID
        #
        # 12
        # 584
        # 1478
        # 999999999
        # -----------------------------------------------------

        if re.fullmatch(r"\d+", text):
            return text

        # -----------------------------------------------------
        # Standalone alphanumeric ID
        #
        # ABC123
        # A12B
        # X-123
        # -----------------------------------------------------

        if re.fullmatch(
            r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*",
            text
        ):
            if any(char.isdigit() for char in text):
                return text

        return None

    # =========================================================
    # REFUND ID
    # =========================================================

    def extract_refund_id(self, text):

        text = text.strip()

        # A numeric ID is valid only when the conversation explicitly asks
        # for a refund ID; the caller selects this extractor in that context.
        if re.fullmatch(r"\d+", text):
            return text

        # -----------------------------------------------------
        # Explicit refund labels
        #
        # REFUND R1001
        # REF R1001
        # REFUND-1478
        # -----------------------------------------------------

        match = re.search(
            r"\b(?:refund|ref|refundid|refund_id)"
            r"[\s:#_-]*([A-Za-z0-9][A-Za-z0-9_-]*)\b",
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1)

        # -----------------------------------------------------
        # Traditional refund ID format
        #
        # R1001
        # R12
        # R999999
        # -----------------------------------------------------

        match = re.fullmatch(
            r"R[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*",
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0).upper()

        return None

    # =========================================================
    # GENERIC ID
    # =========================================================

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
