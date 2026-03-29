"""
Shop4You configuration.
All settings, env vars, and department metadata live here so
the rest of the codebase can just import what it needs.
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# -- Logging --
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(name)-12s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# -- LLM settings --
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o"
LLM_MODEL_FALLBACK = "gpt-4.1"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"

# Tavily key (web search, not required for core flow)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# -- Vector store --
CHROMA_PERSIST_DIR = "./knowledge_base/chroma"
CHROMA_COLLECTION_NAME = "shop4you_kb"

# -- Persistent memory --
SQLITE_MEMORY_DB = "shop4you_memory.db"

# -- Reflection loop --
MAX_REFLECTION_ATTEMPTS = 3

# -- CrewAI escalation --
# When True, escalated queries trigger a 3-agent investigation crew
# (Complaint Analyst -> Account Investigator -> Resolution Specialist).
# Set to False to use the simpler escalation-message-only approach.
CREWAI_ESCALATION_ENABLED = os.getenv("CREWAI_ESCALATION", "true").lower() in ("true", "1", "yes")

# -- RAG retrieval --
RAG_TOP_K = 3
RAG_SCORE_THRESHOLD = 0.2

# -- Data directory --
DATA_DIR = "./data"


# ---------- Department definitions (8 total) ----------
DEPARTMENTS = {
    # Customer-facing departments
    "orders_returns": {
        "name": "Orders & Returns",
        "audience": "external",
        "description": (
            "Handles customer queries about order placement, order tracking, "
            "order cancellations, return and refund policies, exchange processes, "
            "and order-related complaints for Shop4You, a UK-based online retail company."
        ),
        "tone": "Friendly, empathetic, solution-oriented",
        "example_topics": [
            "order tracking", "return policy", "refund status",
            "cancel order", "exchange item", "missing item",
        ],
    },
    "billing_payments": {
        "name": "Billing & Payments",
        "audience": "external",
        "description": (
            "Handles customer queries about payment methods, invoices, billing errors, "
            "refund timelines, promo codes, gift cards, instalment plans, and payment "
            "security for Shop4You, a UK-based online retail company."
        ),
        "tone": "Professional, clear, reassuring",
        "example_topics": [
            "payment methods", "invoice request", "billing error",
            "promo code", "gift card balance", "refund timeline",
        ],
    },
    "shipping_delivery": {
        "name": "Shipping & Delivery",
        "audience": "external",
        "description": (
            "Handles customer queries about shipping options, delivery times, "
            "tracking parcels, international shipping, delivery failures, address "
            "changes, and shipping costs for Shop4You, a UK-based online retail company."
        ),
        "tone": "Helpful, precise, proactive",
        "example_topics": [
            "delivery time", "shipping cost", "track parcel",
            "international shipping", "change address", "failed delivery",
        ],
    },
    "product_inquiries": {
        "name": "Product Inquiries",
        "audience": "external",
        "description": (
            "Handles customer queries about product availability, sizing guides, "
            "product specifications, stock alerts, product comparisons, warranties, "
            "and recommendations for Shop4You, a UK-based online retail company."
        ),
        "tone": "Enthusiastic, knowledgeable, helpful",
        "example_topics": [
            "product availability", "size guide", "product specs",
            "stock alert", "warranty info", "recommendations",
        ],
    },

    # Internal / employee-facing departments
    "hr": {
        "name": "Human Resources (HR)",
        "audience": "internal",
        "description": (
            "Handles employee queries about leave policies, payroll, benefits, "
            "onboarding, performance reviews, workplace policies, training programmes, "
            "and grievance procedures for Shop4You, a UK-based retail company."
        ),
        "tone": "Supportive, professional, confidential",
        "example_topics": [
            "annual leave", "sick leave", "payslip",
            "onboarding", "performance review", "benefits enrolment",
        ],
    },
    "it_helpdesk": {
        "name": "IT Helpdesk",
        "audience": "internal",
        "description": (
            "Handles employee queries about VPN access, password resets, software "
            "installations, hardware requests, email issues, system outages, security "
            "policies, and troubleshooting for Shop4You, a UK-based retail company."
        ),
        "tone": "Technical but accessible, patient, step-by-step",
        "example_topics": [
            "VPN access", "password reset", "software install",
            "hardware request", "email issue", "system outage",
        ],
    },
    "operations": {
        "name": "Operations",
        "audience": "internal",
        "description": (
            "Handles employee queries about warehouse procedures, inventory management, "
            "supply chain updates, shift scheduling, vendor coordination, quality "
            "control, and logistics for Shop4You, a UK-based retail company."
        ),
        "tone": "Concise, factual, process-oriented",
        "example_topics": [
            "shift schedule", "inventory check", "warehouse procedure",
            "vendor contact", "quality control", "supply chain update",
        ],
    },
    "loyalty_programme": {
        "name": "Loyalty Programme & Employee Benefits",
        "audience": "internal",
        "description": (
            "Handles queries about the Shop4You loyalty points system, employee "
            "discount programme, reward redemption, tier upgrades, referral bonuses, "
            "and special employee perks for Shop4You, a UK-based retail company."
        ),
        "tone": "Warm, encouraging, informative",
        "example_topics": [
            "loyalty points", "employee discount", "reward redemption",
            "tier upgrade", "referral bonus", "employee perks",
        ],
    },
}

# Quick-access helpers
EXTERNAL_DEPARTMENTS = [k for k, v in DEPARTMENTS.items() if v["audience"] == "external"]
INTERNAL_DEPARTMENTS = [k for k, v in DEPARTMENTS.items() if v["audience"] == "internal"]
ALL_DEPARTMENT_KEYS = list(DEPARTMENTS.keys())
ALL_DEPARTMENT_NAMES = [v["name"] for v in DEPARTMENTS.values()]


def get_department_by_name(name: str) -> dict | None:
    """Find a department dict by its display name (case-insensitive)."""
    for key, dept in DEPARTMENTS.items():
        if dept["name"].lower() == name.lower():
            return {"key": key, **dept}
    return None


def get_department_keys_for_routing() -> list[str]:
    """All valid department keys plus 'unknown'  --  used by the router."""
    return ALL_DEPARTMENT_KEYS + ["unknown"]
