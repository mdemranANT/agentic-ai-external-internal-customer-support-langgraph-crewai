"""
Dummy order database for Shop4You demo.

Each customer (identified by email) has a handful of orders with
realistic statuses, items, and dates.  The agent's `lookup_orders`
tool queries this data so it can answer questions like
"When did I last order?" or "What's the status of my order?"
"""

from datetime import date

ORDERS: dict[str, list[dict]] = {
    # -- Sarah Ahmed (Gold) -------------------------------------
    "sarah@gmail.com": [
        {
            "order_id": "ORD-4821",
            "date": "2026-02-10",
            "items": [
                {"name": "Running Shoes (Black, Size 7)", "qty": 1, "price": 79.99},
                {"name": "Cotton T-Shirt (White, M)", "qty": 2, "price": 15.00},
            ],
            "total": 109.99,
            "status": "Delivered",
            "tracking": "TRK-UK-882341",
        },
        {
            "order_id": "ORD-4655",
            "date": "2026-01-22",
            "items": [
                {"name": "Leather Wallet (Brown)", "qty": 1, "price": 29.99},
            ],
            "total": 29.99,
            "status": "Delivered",
            "tracking": "TRK-UK-871204",
        },
        {
            "order_id": "ORD-4290",
            "date": "2025-12-15",
            "items": [
                {"name": "Blue Wool Jumper (Navy, L)", "qty": 1, "price": 45.00},
                {"name": "Wireless Headphones (Black)", "qty": 1, "price": 59.99},
            ],
            "total": 104.99,
            "status": "Delivered",
            "tracking": "TRK-UK-859017",
        },
    ],

    # -- James Wilson (Silver) ----------------------------------
    "james@outlook.com": [
        {
            "order_id": "ORD-4910",
            "date": "2026-02-18",
            "items": [
                {"name": "Running Shoes (White, Size 10)", "qty": 1, "price": 79.99},
            ],
            "total": 79.99,
            "status": "Shipped",
            "tracking": "TRK-UK-884521",
        },
        {
            "order_id": "ORD-4712",
            "date": "2026-02-01",
            "items": [
                {"name": "Cotton T-Shirt (Grey, L)", "qty": 3, "price": 15.00},
            ],
            "total": 45.00,
            "status": "Delivered",
            "tracking": "TRK-UK-878102",
        },
    ],

    # -- Emma Chen (Bronze) -------------------------------------
    "emma@yahoo.com": [
        {
            "order_id": "ORD-4870",
            "date": "2026-02-14",
            "items": [
                {"name": "Blue Wool Jumper (Navy, S)", "qty": 1, "price": 45.00},
                {"name": "Leather Wallet (Black)", "qty": 1, "price": 29.99},
            ],
            "total": 74.99,
            "status": "Processing",
            "tracking": None,
        },
        {
            "order_id": "ORD-4501",
            "date": "2026-01-05",
            "items": [
                {"name": "Cotton T-Shirt (Navy, XS)", "qty": 1, "price": 15.00},
            ],
            "total": 15.00,
            "status": "Delivered",
            "tracking": "TRK-UK-874330",
        },
    ],

    # -- Ali Khan (Standard) ------------------------------------
    "ali@hotmail.com": [
        {
            "order_id": "ORD-4935",
            "date": "2026-02-20",
            "items": [
                {"name": "Wireless Headphones (Black)", "qty": 1, "price": 59.99},
                {"name": "Cotton T-Shirt (Black, XL)", "qty": 1, "price": 15.00},
            ],
            "total": 74.99,
            "status": "Shipped",
            "tracking": "TRK-UK-885102",
        },
    ],

    # -- Maria Lopez (Gold) -------------------------------------
    "maria@gmail.com": [
        {
            "order_id": "ORD-4880",
            "date": "2026-02-15",
            "items": [
                {"name": "Running Shoes (Red, Size 6)", "qty": 1, "price": 79.99},
                {"name": "Blue Wool Jumper (Navy, M)", "qty": 1, "price": 45.00},
            ],
            "total": 124.99,
            "status": "Delivered",
            "tracking": "TRK-UK-883710",
        },
        {
            "order_id": "ORD-4620",
            "date": "2026-01-18",
            "items": [
                {"name": "Leather Wallet (Brown)", "qty": 2, "price": 29.99},
            ],
            "total": 59.98,
            "status": "Delivered",
            "tracking": "TRK-UK-876541",
        },
        {
            "order_id": "ORD-4350",
            "date": "2025-12-28",
            "items": [
                {"name": "Cotton T-Shirt (White, S)", "qty": 4, "price": 15.00},
            ],
            "total": 60.00,
            "status": "Delivered",
            "tracking": "TRK-UK-862890",
        },
    ],
}


def get_orders(email: str) -> list[dict]:
    """Return all orders for a given email (case-insensitive)."""
    return ORDERS.get(email.strip().lower(), [])


def get_order_by_id(order_id: str) -> dict | None:
    """Find a specific order across all customers."""
    order_id = order_id.strip().upper()
    for orders in ORDERS.values():
        for order in orders:
            if order["order_id"] == order_id:
                return order
    return None


def format_orders_summary(email: str) -> str:
    """Return a human-readable summary of a customer's orders."""
    orders = get_orders(email)
    if not orders:
        return f"No orders found for {email}."

    lines = [f"Orders for {email} ({len(orders)} total):\n"]
    for o in orders:
        items_str = ", ".join(
            f"{i['name']} (x{i['qty']})" for i in o["items"]
        )
        tracking = f"Tracking: {o['tracking']}" if o["tracking"] else "Tracking: Pending"
        lines.append(
            f"  {o['order_id']} | {o['date']} | {o['status']} | "
            f"GBP {o['total']:.2f}\n"
            f"    Items: {items_str}\n"
            f"    {tracking}"
        )
    return "\n".join(lines)


def format_single_order(order: dict) -> str:
    """Format one order into a readable string."""
    items_str = ", ".join(
        f"{i['name']} (x{i['qty']}, GBP {i['price']:.2f})" for i in order["items"]
    )
    tracking = f"Tracking: {order['tracking']}" if order["tracking"] else "Tracking: Pending"
    return (
        f"Order {order['order_id']}  --  {order['date']}\n"
        f"Status: {order['status']}\n"
        f"Items: {items_str}\n"
        f"Total: GBP {order['total']:.2f}\n"
        f"{tracking}"
    )
