"""
Tool functions the agent can call.
RAG retrieval, department listing, escalation, product search, loyalty lookup,
and order lookup.
"""
from langchain_core.tools import tool

from config import DEPARTMENTS, ALL_DEPARTMENT_KEYS
from vector_store import create_vector_store, retrieve_context
from orders_db import get_orders, get_order_by_id, format_orders_summary, format_single_order


# We only create the vector store once and reuse it after that.
_vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store()
    return _vector_store


# ---------- RAG retrieval ----------
@tool
def search_knowledge_base(query: str, department_key: str) -> str:
    """Search the Shop4You knowledge base for a specific department.
    Use this to find answers to customer or employee questions.
    Args:
        query: The user's question.
        department_key: One of: orders_returns, billing_payments, shipping_delivery,
                        product_inquiries, hr, it_helpdesk, operations, loyalty_programme.
    Returns:
        Relevant FAQ content from the department's knowledge base, or a
        message saying no results were found.
    """
    if department_key not in ALL_DEPARTMENT_KEYS:
        return f"Unknown department '{department_key}'. Valid: {ALL_DEPARTMENT_KEYS}"

    vs = _get_vector_store()
    context, docs = retrieve_context(vs, query, department_key)

    if not context:
        dept_name = DEPARTMENTS[department_key]["name"]
        return f"No relevant information found in the {dept_name} knowledge base for this query."

    return context


# ---------- Department listing ----------
@tool
def list_departments() -> str:
    """List all available Shop4You departments and their descriptions.
    Use this when the user asks what departments or services are available.
    Returns:
        A formatted list of all departments with audience type and description.
    """
    lines = []
    for key, dept in DEPARTMENTS.items():
        audience_label = "Customer" if dept["audience"] == "external" else "Employee"
        lines.append(f"- {dept['name']} ({audience_label}): {dept['description'][:100]}...")
    return "\n".join(lines)


# ---------- Escalation ----------
@tool
def escalate_to_human(reason: str, customer_name: str = "", customer_email: str = "", customer_phone: str = "") -> str:
    """Escalate the query to a human support agent.
    Use this when the customer is upset, the query cannot be answered by AI,
    or the department is unknown.
    Args:
        reason: Why the query is being escalated.
        customer_name: Customer's name (if provided).
        customer_email: Customer's email (if provided).
        customer_phone: Customer's phone number (if provided).
    Returns:
        Confirmation message that escalation has been registered.
    """
    details = []
    if customer_name:
        details.append(f"Name: {customer_name}")
    if customer_email:
        details.append(f"Email: {customer_email}")
    if customer_phone:
        details.append(f"Phone: {customer_phone}")

    contact_info = "\n".join(details) if details else "No contact details provided."

    return (
        f"[PASS] ESCALATION REGISTERED\n"
        f"Reason: {reason}\n"
        f"Contact Details:\n{contact_info}\n\n"
        f"A human support agent will reach out to you shortly. "
        f"Your reference number is ESC-{hash(reason) % 100000:05d}."
    )


# ---------- Product search (stretch goal) ----------
@tool
def search_product(product_name: str) -> str:
    """Search for product availability and details at Shop4You.
    Args:
        product_name: Name or description of the product to search for.
    Returns:
        Product availability status and details.
    """
    # Hardcoded catalogue for demo
    products = {
        "blue wool jumper": {"status": "In Stock", "price": "GBP 45.00", "sizes": "S, M, L, XL", "colour": "Navy Blue"},
        "running shoes": {"status": "In Stock", "price": "GBP 79.99", "sizes": "6-12", "colour": "Black, White, Red"},
        "leather wallet": {"status": "Low Stock", "price": "GBP 29.99", "sizes": "One Size", "colour": "Brown, Black"},
        "wireless headphones": {"status": "Out of Stock", "price": "GBP 59.99", "sizes": "One Size", "colour": "Black"},
        "cotton t-shirt": {"status": "In Stock", "price": "GBP 15.00", "sizes": "XS-XXL", "colour": "White, Black, Grey, Navy"},
    }

    search_lower = product_name.lower()
    for name, info in products.items():
        if name in search_lower or search_lower in name:
            return (
                f"Product: {name.title()}\n"
                f"Status: {info['status']}\n"
                f"Price: {info['price']}\n"
                f"Sizes: {info['sizes']}\n"
                f"Colours: {info['colour']}"
            )

    return f"No exact match found for '{product_name}'. Please check our website for the full catalogue or try a different search term."


# ---------- Loyalty points lookup (stretch goal) ----------
@tool
def check_loyalty_points(employee_id: str) -> str:
    """Check loyalty points balance and tier for a Shop4You employee.
    Args:
        employee_id: The employee's ID (e.g., EMP001).
    Returns:
        Points balance, tier, and available rewards.
    """
    # Hardcoded employee data for demo
    mock_data = {
        "EMP001": {"name": "Alice Smith", "points": 2450, "tier": "Gold", "discount": "15%"},
        "EMP002": {"name": "Bob Johnson", "points": 890, "tier": "Silver", "discount": "10%"},
        "EMP003": {"name": "Carol Williams", "points": 5200, "tier": "Platinum", "discount": "20%"},
    }

    employee_id_upper = employee_id.upper()
    if employee_id_upper in mock_data:
        data = mock_data[employee_id_upper]
        return (
            f"Employee: {data['name']} ({employee_id_upper})\n"
            f"Loyalty Tier: {data['tier']}\n"
            f"Points Balance: {data['points']:,}\n"
            f"Staff Discount: {data['discount']}\n"
            f"Available Rewards: Gift vouchers, extra day off, partner discounts"
        )

    return f"Employee ID '{employee_id}' not found. Please check and try again."


# ---------- Order lookup ----------
@tool
def lookup_orders(customer_email: str, order_id: str = "") -> str:
    """Look up order history or a specific order for a Shop4You customer.
    Use this when the customer asks about their orders, delivery status,
    returns, or recent purchases.
    Args:
        customer_email: The customer's email address.
        order_id: (Optional) A specific order ID like ORD-4821.  If empty,
                  return the full order history.
    Returns:
        Order details or a summary of all orders for the customer.
    """
    if order_id:
        order = get_order_by_id(order_id)
        if order:
            return format_single_order(order)
        return f"Order '{order_id}' not found. Please double-check the order number."

    return format_orders_summary(customer_email)


# Expose all tools as a flat list so the agent file can import them easily
ALL_TOOLS = [
    search_knowledge_base,
    list_departments,
    escalate_to_human,
    search_product,
    check_loyalty_points,
    lookup_orders,
]
