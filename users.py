"""
Simple user registry for Shop4You.

Users are identified by email.
  - @shop4you.com  -> employee  (sees internal departments)
  - anything else  -> customer  (sees external departments)

The USERS dict holds a handful of demo users. Unknown emails are
auto-registered as customers with a generic profile.
"""

USERS: dict[str, dict] = {
    # -- Customers ----------------------------------------------
    "sarah@gmail.com": {
        "name": "Sarah Ahmed",
        "type": "customer",
        "tier": "Gold",
    },
    "james@outlook.com": {
        "name": "James Wilson",
        "type": "customer",
        "tier": "Silver",
    },
    "emma@yahoo.com": {
        "name": "Emma Chen",
        "type": "customer",
        "tier": "Bronze",
    },
    "ali@hotmail.com": {
        "name": "Ali Khan",
        "type": "customer",
        "tier": "Standard",
    },
    "maria@gmail.com": {
        "name": "Maria Lopez",
        "type": "customer",
        "tier": "Gold",
    },

    # -- Employees ----------------------------------------------
    "john@shop4you.com": {
        "name": "John Davies",
        "type": "employee",
        "department": "Operations",
    },
    "priya@shop4you.com": {
        "name": "Priya Sharma",
        "type": "employee",
        "department": "HR",
    },
    "mike@shop4you.com": {
        "name": "Mike Thompson",
        "type": "employee",
        "department": "IT Helpdesk",
    },
    "fatima@shop4you.com": {
        "name": "Fatima Noor",
        "type": "employee",
        "department": "Loyalty Programme",
    },
    "david@shop4you.com": {
        "name": "David Roberts",
        "type": "employee",
        "department": "Operations",
    },
}


def get_user(email: str) -> dict:
    """
    Look up a user by email (case-insensitive).
    Unknown emails are auto-registered as customers.
    """
    email = email.strip().lower()
    if email in USERS:
        return {"email": email, **USERS[email]}

    # Auto-register: employee if @shop4you.com, else customer
    if email.endswith("@shop4you.com"):
        name = email.split("@")[0].title()
        return {"email": email, "name": name, "type": "employee", "department": "General"}
    else:
        name = email.split("@")[0].title()
        return {"email": email, "name": name, "type": "customer", "tier": "Standard"}


def is_employee(email: str) -> bool:
    """Quick check if the email belongs to an employee."""
    return get_user(email)["type"] == "employee"
