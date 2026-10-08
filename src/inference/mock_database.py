ORDERS = {

    "12": {
        "order_id": "12",
        "status": "Processing",
        "item": "Demo Product",
        "delivery_date": "2026-10-10",
        "amount": 999,
    },

    "584": {
        "order_id": "584",
        "status": "Processing",
        "item": "Bluetooth Speaker",
        "delivery_date": "2026-10-08",
        "amount": 1999,
    },

    "1478": {
        "order_id": "1478",
        "status": "Shipped",
        "item": "Smart Watch",
        "delivery_date": "2026-10-09",
        "amount": 3999,
    },

    "12345": {
        "order_id": "12345",
        "status": "Shipped",
        "item": "Wireless Headphones",
        "delivery_date": "2026-10-07",
        "amount": 2499,
    },

    "12346": {
        "order_id": "12346",
        "status": "Processing",
        "item": "Smart Watch",
        "delivery_date": "2026-10-09",
        "amount": 3999,
    },

    "12347": {
        "order_id": "12347",
        "status": "Delivered",
        "item": "Bluetooth Speaker",
        "delivery_date": "2026-10-02",
        "amount": 1999,
    },
}


REFUNDS = {

    "1": {
        "refund_id": "1",
        "order_id": "12",
        "status": "Processing",
        "amount": 999,
    },

    "584": {
        "refund_id": "584",
        "order_id": "584",
        "status": "Processing",
        "amount": 1999,
    },

    "R1001": {
        "refund_id": "R1001",
        "order_id": "12345",
        "status": "Processing",
        "amount": 2499,
    },

    "R1002": {
        "refund_id": "R1002",
        "order_id": "12346",
        "status": "Completed",
        "amount": 3999,
    },
}


INVOICES = {

    "1": {
        "invoice_id": "1",
        "order_id": "12",
    },

    "584": {
        "invoice_id": "584",
        "order_id": "584",
    },

    "INV12345": {
        "invoice_id": "INV12345",
        "order_id": "12345",
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

    return ORDERS.get(
        str(order_id)
    )


def get_refund(refund_id):
    if refund_id is None:
        return None

    return REFUNDS.get(
        str(refund_id)
    )


def get_invoice(invoice_id):
    if invoice_id is None:
        return None

    return INVOICES.get(
        str(invoice_id)
    )


def get_user(user_id):
    if user_id is None:
        return None

    return USERS.get(
        str(user_id)
    )