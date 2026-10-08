from .mock_database import (
    get_order,
    get_refund,
    get_invoice,
)


class WorkflowHandler:

    # =========================================================
    # MAIN DISPATCHER
    # =========================================================

    def handle(self, intent, context):

        handlers = {

            "track_order":
                lambda: self.track_order(context),

            "place_order":
                self.place_order,

            "change_order":
                lambda: self.change_order(context),

            "cancel_order":
                lambda: self.cancel_order(context),

            "change_shipping_address":
                lambda: self.change_shipping_address(context),

            "set_up_shipping_address":
                self.set_up_shipping_address,

            "delivery_options":
                self.delivery_options,

            "delivery_period":
                self.delivery_period,

            "get_refund":
                lambda: self.get_refund(context),

            "track_refund":
                lambda: self.track_refund(context),

            "check_refund_policy":
                self.check_refund_policy,

            "check_invoice":
                lambda: self.check_invoice(context),

            "get_invoice":
                lambda: self.get_invoice(context),

            "payment_issue":
                self.payment_issue,

            "check_payment_methods":
                self.check_payment_methods,

            "recover_password":
                self.recover_password,

            "create_account":
                self.create_account,

            "delete_account":
                self.delete_account,

            "edit_account":
                self.edit_account,

            "switch_account":
                self.switch_account,

            "check_cancellation_fee":
                self.check_cancellation_fee,

            "complaint":
                self.complaint,

            "contact_customer_service":
                self.contact_customer_service,

            "contact_human_agent":
                self.contact_human_agent,

            "newsletter_subscription":
                self.newsletter_subscription,

            "registration_problems":
                self.registration_problems,

            "review":
                self.review,
        }

        handler = handlers.get(intent)

        if not handler:

            return {
                "status": "unsupported",
                "message": (
                    "I am not able to handle this request yet."
                )
            }

        return handler()

    # =========================================================
    # CONFIRMATION EXECUTOR
    # =========================================================

    def confirm(self, intent, context):

        handlers = {

            "cancel_order":
                lambda: self._confirm_cancel_order(context),

            "get_refund":
                lambda: self._confirm_get_refund(context),

            "delete_account":
                self._confirm_delete_account,

            "change_order":
                lambda: self._confirm_change_order(context),

            "change_shipping_address":
                lambda: self._confirm_change_shipping_address(context),
        }

        handler = handlers.get(intent)

        if not handler:

            return {
                "status": "success",
                "message": (
                    "Your request has been confirmed."
                )
            }

        return handler()

    # =========================================================
    # TRACK ORDER
    # =========================================================

    def track_order(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        if not order_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your order number."
                )
            }

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        return {
            "status": "success",
            "message": (
                f"Your order {order_id} is currently "
                f"{order['status']}. "
                f"Your {order['item']} is expected to arrive by "
                f"{order['delivery_date']}."
            ),
            "data": order,
        }

    # =========================================================
    # PLACE ORDER
    # =========================================================

    def place_order(self):

        return {
            "status": "success",
            "message": (
                "You can place an order by selecting a product "
                "and completing checkout."
            )
        }

    # =========================================================
    # CHANGE ORDER
    # =========================================================

    def change_order(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        if not order_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your order number."
                )
            }

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        if order["status"] in {
            "Shipped",
            "Delivered",
        }:

            return {
                "status": "cannot_change",
                "message": (
                    f"Order {order_id} has already been "
                    f"{order['status'].lower()} "
                    "and cannot be modified."
                ),
                "data": order,
            }

        return {
            "status": "needs_confirmation",
            "requires_confirmation": True,
            "message": (
                f"Order {order_id} can be modified. "
                "Please confirm that you want to modify it."
            ),
            "data": order,
        }

    def _confirm_change_order(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        return {
            "status": "success",
            "message": (
                f"Order {order_id} modification has been confirmed. "
                "Please provide the changes you want to make."
            ),
            "data": {
                "order_id": order_id
            },
        }

    # =========================================================
    # CANCEL ORDER
    # =========================================================

    def cancel_order(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        if not order_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your order number."
                )
            }

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        if order["status"] in {
            "Shipped",
            "Delivered",
        }:

            return {
                "status": "cannot_cancel",
                "message": (
                    f"Order {order_id} has already been "
                    f"{order['status'].lower()} "
                    "and cannot be cancelled."
                ),
                "data": order,
            }

        return {
            "status": "needs_confirmation",
            "requires_confirmation": True,
            "message": (
                f"Order {order_id} can be cancelled. "
                "Please confirm that you want to cancel it."
            ),
            "data": order,
        }

    def _confirm_cancel_order(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        # Simulate state change in mock database.
        order["status"] = "Cancelled"

        return {
            "status": "success",
            "message": (
                f"Order {order_id} cancellation has been confirmed."
            ),
            "data": order,
        }

    # =========================================================
    # SHIPPING ADDRESS
    # =========================================================

    def change_shipping_address(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        if not order_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your order number."
                )
            }

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        if order["status"] in {
            "Shipped",
            "Delivered",
        }:

            return {
                "status": "cannot_change",
                "message": (
                    f"The shipping address for order "
                    f"{order_id} cannot be changed because "
                    f"the order has already been "
                    f"{order['status'].lower()}."
                ),
                "data": order,
            }

        return {
            "status": "needs_confirmation",
            "requires_confirmation": True,
            "message": (
                f"The shipping address for order {order_id} "
                "can be changed. Please confirm."
            ),
            "data": order,
        }

    def _confirm_change_shipping_address(
        self,
        context
    ):

        order_id = self._get_id(
            context,
            "order_id"
        )

        return {
            "status": "success",
            "message": (
                f"Shipping address change for order "
                f"{order_id} has been confirmed. "
                "Please provide the new address."
            ),
            "data": {
                "order_id": order_id
            },
        }

    def set_up_shipping_address(self):

        return {
            "status": "success",
            "message": (
                "You can add a shipping address from your "
                "account before placing an order."
            )
        }

    # =========================================================
    # DELIVERY
    # =========================================================

    def delivery_options(self):

        return {
            "status": "success",
            "message": (
                "Available delivery options may include "
                "standard and express delivery depending "
                "on your location and order."
            )
        }

    def delivery_period(self):

        return {
            "status": "success",
            "message": (
                "Standard delivery usually takes a few "
                "business days. The exact delivery date "
                "is shown during checkout."
            )
        }

    # =========================================================
    # REFUND
    # =========================================================

    def get_refund(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        if not order_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your order number."
                )
            }

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        return {
            "status": "needs_confirmation",
            "requires_confirmation": True,
            "message": (
                f"You can request a refund for order {order_id}. "
                "Please confirm that you want to proceed."
            ),
            "data": order,
        }

    def _confirm_get_refund(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        # Generate a demo refund ID.
        refund_id = f"REF-{order_id}"

        return {
            "status": "success",
            "message": (
                f"Your refund request for order {order_id} "
                f"has been confirmed. "
                f"Refund ID: {refund_id}."
            ),
            "data": {
                "refund_id": refund_id,
                "order_id": order_id,
                "status": "Processing",
                "amount": order.get(
                    "amount",
                    0
                ),
            },
        }

    # =========================================================
    # TRACK REFUND
    # =========================================================

    def track_refund(self, context):

        refund_id = self._get_id(
            context,
            "refund_id"
        )

        if not refund_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your refund ID."
                )
            }

        refund = get_refund(refund_id)

        if not refund:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find refund {refund_id}."
                )
            }

        return {
            "status": "success",
            "message": (
                f"Refund {refund_id} is currently "
                f"{refund['status']}. "
                f"The refund amount is ₹"
                f"{refund['amount']}."
            ),
            "data": refund,
        }

    def check_refund_policy(self):

        return {
            "status": "success",
            "message": (
                "Refunds are available for eligible orders. "
                "Eligibility depends on the order status "
                "and purchase conditions."
            )
        }

    # =========================================================
    # INVOICE
    # =========================================================

    def check_invoice(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        if not order_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your order number."
                )
            }

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        invoice = self._find_invoice_for_order(
            order_id
        )

        if invoice:

            return {
                "status": "success",
                "message": (
                    f"An invoice is available for "
                    f"order {order_id}."
                ),
                "data": invoice,
            }

        return {
            "status": "success",
            "message": (
                f"An invoice is available for "
                f"order {order_id}."
            ),
            "data": order,
        }

    def get_invoice(self, context):

        order_id = self._get_id(
            context,
            "order_id"
        )

        if not order_id:

            return {
                "status": "need_information",
                "message": (
                    "Please provide your order number."
                )
            }

        order = get_order(order_id)

        if not order:

            return {
                "status": "not_found",
                "message": (
                    f"I could not find order {order_id}."
                )
            }

        invoice = self._find_invoice_for_order(
            order_id
        )

        if invoice:

            return {
                "status": "success",
                "message": (
                    f"Invoice {invoice['invoice_id']} "
                    f"is available for order {order_id}."
                ),
                "data": invoice,
            }

        return {
            "status": "success",
            "message": (
                f"Invoice information for order "
                f"{order_id} is available."
            ),
            "data": {
                "order_id": order_id
            },
        }

    def _find_invoice_for_order(
        self,
        order_id
    ):

        for invoice_id in [
            "1",
            "584",
            "INV12345",
        ]:

            invoice = get_invoice(
                invoice_id
            )

            if (
                invoice
                and str(
                    invoice.get("order_id")
                ) == str(order_id)
            ):

                return invoice

        return None

    # =========================================================
    # PAYMENT
    # =========================================================

    def payment_issue(self):

        return {
            "status": "success",
            "message": (
                "I'm sorry you're having trouble with "
                "your payment. Please check your payment "
                "details and try again."
            )
        }

    def check_payment_methods(self):

        return {
            "status": "success",
            "message": (
                "Available payment methods may include "
                "cards, online banking and other supported "
                "options shown during checkout."
            )
        }

    # =========================================================
    # ACCOUNT
    # =========================================================

    def recover_password(self):

        return {
            "status": "success",
            "message": (
                "You can reset your password using the "
                "password reset option. A reset link will "
                "be sent to your registered email address."
            )
        }

    def create_account(self):

        return {
            "status": "success",
            "message": (
                "You can create a new account using your "
                "email address and completing the registration "
                "process."
            )
        }

    def delete_account(self):

        return {
            "status": "needs_confirmation",
            "requires_confirmation": True,
            "message": (
                "Account deletion is permanent. "
                "Please confirm that you want to delete "
                "your account."
            ),
        }

    def _confirm_delete_account(self):

        return {
            "status": "success",
            "message": (
                "Your account deletion request has been confirmed."
            ),
        }

    def edit_account(self):

        return {
            "status": "success",
            "message": (
                "You can update your name, email address, "
                "phone number and other profile details."
            )
        }

    def switch_account(self):

        return {
            "status": "success",
            "message": (
                "You can switch accounts by signing out "
                "of the current account and signing in "
                "with another account."
            )
        }

    # =========================================================
    # OTHER INTENTS
    # =========================================================

    def check_cancellation_fee(self):

        return {
            "status": "success",
            "message": (
                "Cancellation fees depend on the order "
                "status and applicable purchase conditions."
            )
        }

    def complaint(self):

        return {
            "status": "success",
            "message": (
                "I'm sorry that you had a problem. "
                "Please describe your complaint and I will "
                "help you with the next step."
            )
        }

    def contact_customer_service(self):

        return {
            "status": "success",
            "message": (
                "Customer support is available to help "
                "you with your request."
            )
        }

    def contact_human_agent(self):

        return {
            "status": "success",
            "message": (
                "I will connect you with a human support agent."
            )
        }

    def newsletter_subscription(self):

        return {
            "status": "success",
            "message": (
                "You can manage your newsletter subscription "
                "from your notification preferences."
            )
        }

    def registration_problems(self):

        return {
            "status": "success",
            "message": (
                "If registration is failing, please check "
                "that your email is valid and all required "
                "fields are completed."
            )
        }

    def review(self):

        return {
            "status": "success",
            "message": (
                "We would be happy to receive your feedback. "
                "Please share your review."
            )
        }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _get_id(
        context,
        key
    ):

        value = context.get(key)

        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value