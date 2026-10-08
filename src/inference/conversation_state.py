class ConversationState:

    def __init__(self):
        self.history = []

        self.current_intent = None
        self.current_category = None

        self.context = {}

        self.awaiting_clarification = False
        self.awaiting_entity = None

        self.awaiting_confirmation = False
        self.confirmation_action = None

    # =========================================================
    # HISTORY
    # =========================================================

    def add_message(self, role, message):
        self.history.append({
            "role": role,
            "message": message
        })

    def get_history(self):
        return self.history

    # =========================================================
    # INTENT
    # =========================================================

    def set_intent(self, intent, category=None):
        self.current_intent = intent
        self.current_category = category

    # =========================================================
    # CONTEXT
    # =========================================================

    def set_context(self, key, value):
        self.context[key] = value

    def get_context(self, key, default=None):
        return self.context.get(key, default)

    # =========================================================
    # ENTITY WAITING
    # =========================================================

    def set_expected_entity(self, entity):
        self.awaiting_entity = entity

    def get_expected_entity(self):
        return self.awaiting_entity

    def is_awaiting_entity(self):
        return self.awaiting_entity is not None

    def clear_expected_entity(self):
        self.awaiting_entity = None

    # =========================================================
    # CLARIFICATION
    # =========================================================

    def set_clarification(self, value):
        self.awaiting_clarification = value

    def is_awaiting_clarification(self):
        return self.awaiting_clarification

    # =========================================================
    # CONFIRMATION
    # =========================================================

    def set_confirmation(self, value, action=None):
        self.awaiting_confirmation = value
        self.confirmation_action = action if value else None

    def is_awaiting_confirmation(self):
        return self.awaiting_confirmation

    def get_confirmation_action(self):
        return self.confirmation_action

    def clear_confirmation(self):
        self.awaiting_confirmation = False
        self.confirmation_action = None

    # =========================================================
    # RESET
    # =========================================================

    def clear(self):
        self.history = []

        self.current_intent = None
        self.current_category = None

        self.context = {}

        self.awaiting_clarification = False
        self.awaiting_entity = None

        self.awaiting_confirmation = False
        self.confirmation_action = None