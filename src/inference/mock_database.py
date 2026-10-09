ORDERS = {

    "101": {
        "order_id": "101",
        "status": "Processing",
        "item": "Demo Product",
        "delivery_date": "2026-10-10",
        "amount": 999,
    },

    "102": {
        "order_id": "102",
        "status": "Processing",
        "item": "Bluetooth Speaker",
        "delivery_date": "2026-10-08",
        "amount": 1999,
    },

    "103": {
        "order_id": "103",
        "status": "Shipped",
        "item": "Smart Watch",
        "delivery_date": "2026-10-09",
        "amount": 3999,
    },

    "104": {
        "order_id": "104",
        "status": "Shipped",
        "item": "Wireless Headphones",
        "delivery_date": "2026-10-07",
        "amount": 2499,
    },

    "105": {
        "order_id": "105",
        "status": "Processing",
        "item": "Smart Watch",
        "delivery_date": "2026-10-09",
        "amount": 3999,
    },

    "106": {
        "order_id": "106",
        "status": "Delivered",
        "item": "Bluetooth Speaker",
        "delivery_date": "2026-10-02",
        "amount": 1999,
    },
}


REFUNDS = {

    "1": {
        "refund_id": "1",
        "order_id": "101",
        "status": "Processing",
        "amount": 999,
    },

    "2": {
        "refund_id": "2",
        "order_id": "102",
        "status": "Processing",
        "amount": 1999,
    },

    "3": {
        "refund_id": "3",
        "order_id": "103",
        "status": "Processing",
        "amount": 2499,
    },

    "4": {
        "refund_id": "4",
        "order_id": "104",
        "status": "Completed",
        "amount": 3999,
    },
}


INVOICES = {

    "1": {
        "invoice_id": "1",
        "order_id": "101",
    },

    "2": {
        "invoice_id": "2",
        "order_id": "102",
    },

    "3": {
        "invoice_id": "3",
        "order_id": "103",
    },
}


USERS = {

    "user001": {
        "user_id": "user001",
        "name": "Customer",
        "email": "customer@example.com",
        "phone": "+919999999999",
        "account_status": "Active",
    }
}


def get_order(order_id):
    if order_id is None:
        return None
    return ORDERS.get(str(order_id))


def get_refund(refund_id):
    if refund_id is None:
        return None
    return REFUNDS.get(str(refund_id))


def get_invoice(invoice_id):
    if invoice_id is None:
        return None
    return INVOICES.get(str(invoice_id))

def get_user(user_id):
    if user_id is None:
        return None
    return USERS.get(str(user_id))