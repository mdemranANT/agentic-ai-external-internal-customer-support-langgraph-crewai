"""
Shop4You  --  Streamlit Chat Interface

A clean, ChatGPT-style chat UI that:
  1. Asks the user for their email
  2. Identifies them as customer or employee
  3. Opens a persistent chat session powered by the LangGraph agent
  4. Stores conversation memory across sessions (SqliteSaver)

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
from pathlib import Path

# -- Page config (must be first Streamlit call) ------------------------
st.set_page_config(
    page_title="Shop4You AI Assistant",
    page_icon="\U0001f6d2",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -- Custom CSS for a clean chat look ----------------------------------
st.markdown("""
<style>
    /* Hide default Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chat container spacing */
    .stChatMessage {margin-bottom: 0.5rem;}

    /* Login card */
    .login-card {
        max-width: 420px;
        margin: 4rem auto;
        padding: 2.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    .login-card h2 {color: white; margin-bottom: 0.3rem;}
    .login-card p {color: rgba(255,255,255,0.8); font-size: 0.95rem;}

    /* Sidebar user card */
    .user-badge {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .user-badge h4 {margin: 0; color: white;}
    .user-badge p {margin: 0.2rem 0 0 0; color: rgba(255,255,255,0.85); font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


# -- Lazy-load heavy imports (only once) -------------------------------
@st.cache_resource(show_spinner="Loading Shop4You AI Agent...")
def load_agent():
    """Compile the LangGraph agent once and cache it."""
    from agents import compile_graph
    return compile_graph(use_memory=True, persist=True)


def get_user_info(email: str) -> dict:
    """Look up user by email."""
    from users import get_user
    return get_user(email)


def query_agent(agent, query: str, user_id: str, user_context: str = "") -> dict:
    """Send a query through the agent."""
    from agents import run_query
    return run_query(query, user_id=user_id, agent=agent, user_context=user_context)


def build_user_context(user: dict) -> str:
    """Build a context string so the LLM knows who is asking."""
    parts = [f"Name: {user['name']}", f"Email: {user['email']}"]
    if user["type"] == "employee":
        parts.append(f"Role: Employee ({user.get('department', 'N/A')})")
    else:
        parts.append(f"Role: Customer (Loyalty Tier: {user.get('tier', 'Standard')})")
    return "[User Context] " + " | ".join(parts)


# -- Session state defaults --------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_transfer" not in st.session_state:
    st.session_state.pending_transfer = None  # {"query": ..., "target_dept": ...}


# =====================================================================
#  LOGIN SCREEN
# =====================================================================
def show_login():
    """Email login screen  --  identifies customer vs employee."""

    # Logo
    logo_path = Path(__file__).parent / "logo.svg"
    if logo_path.exists():
        st.image(str(logo_path), width=320)

    st.markdown("")  # spacer

    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0.2rem;'>Welcome to Shop4You</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray; margin-bottom:2rem;'>"
        "AI-Powered Customer & Employee Support</p>",
        unsafe_allow_html=True,
    )

    # Email input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        email = st.text_input(
            "Enter your email to start",
            placeholder="e.g. sarah@gmail.com",
            label_visibility="visible",
        )

        st.markdown(
            "<p style='font-size:0.8rem; color:gray; margin-top:-0.5rem;'>"
            "Employees: use your @shop4you.com email</p>",
            unsafe_allow_html=True,
        )

        if st.button("Start Chat", use_container_width=True, type="primary"):
            if not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                user = get_user_info(email)
                st.session_state.user = user
                st.session_state.authenticated = True
                st.session_state.chat_history = []
                st.rerun()

    # Demo accounts hint
    with st.expander("Demo accounts you can try"):
        st.markdown("""
        **Customers:**
        - `sarah@gmail.com`  --  Sarah Ahmed (Gold tier)
        - `james@outlook.com`  --  James Wilson (Silver tier)
        - `ali@hotmail.com`  --  Ali Khan (Standard tier)

        **Employees:**
        - `priya@shop4you.com`  --  Priya Sharma (HR)
        - `mike@shop4you.com`  --  Mike Thompson (IT)
        - `fatima@shop4you.com`  --  Fatima Noor (Loyalty)

        *Any email works  --  unknown emails are auto-registered.*
        """)


# =====================================================================
#  CHAT SCREEN
# =====================================================================
def show_chat():
    """Main chat interface  --  ChatGPT-style."""

    user = st.session_state.user
    is_emp = user["type"] == "employee"

    # -- Sidebar: user info + controls -----------------------------
    with st.sidebar:
        logo_path = Path(__file__).parent / "logo.svg"
        if logo_path.exists():
            st.image(str(logo_path), width=200)

        st.markdown("---")

        # User badge
        badge_extra = (
            f"Dept: {user.get('department', 'N/A')}"
            if is_emp
            else f"Tier: {user.get('tier', 'Standard')}"
        )
        st.markdown(
            f"""<div class="user-badge">
                <h4>{'\U0001f454' if is_emp else '\U0001f6d2'} {user['name']}</h4>
                <p>{user['email']}</p>
                <p>{'Employee' if is_emp else 'Customer'} &bull; {badge_extra}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Department access info
        if is_emp:
            st.caption("You can ask about: HR, IT, Operations, Loyalty, and all customer topics.")
        else:
            st.caption("You can ask about: Orders, Billing, Shipping, Products, and more.")

        st.markdown("---")

        # New chat button
        if st.button("\U0001f5e8 New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # Logout button
        if st.button("\U0001f6aa Logout / Switch User", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.chat_history = []
            st.rerun()

    # -- Chat header -----------------------------------------------
    st.markdown(
        f"### \U0001f6d2 Shop4You AI Assistant"
    )
    st.caption(
        f"Logged in as **{user['name']}** "
        f"({'Employee' if is_emp else 'Customer'})"
    )

    # -- Display chat history --------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="\U0001f9d1" if msg["role"] == "user" else "\U0001f916"):
            st.markdown(msg["content"])

            # Show metadata for assistant messages
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                cols = st.columns(5)
                cols[0].caption(f"Dept: {meta.get('department', ' -- ')}")
                cols[1].caption(f"Sentiment: {meta.get('sentiment', ' -- ')}")
                cols[2].caption(f"Severity: {meta.get('severity', ' -- ')}")
                cols[3].caption(f"Quality: {meta.get('quality_score', ' -- ')}/10")
                cols[4].caption(f"Escalated: {'Yes' if meta.get('escalated') else 'No'}")

                # CrewAI report
                if meta.get("crew_report"):
                    with st.expander("View CrewAI Investigation Report"):
                        crew = meta["crew_report"]
                        for section in ("analysis", "investigation", "resolution"):
                            if crew.get(section):
                                st.markdown(f"**{section.upper()}**")
                                st.write(crew[section][:800])

    # -- Handle pending cross-department transfer ------------------
    if st.session_state.pending_transfer:
        transfer = st.session_state.pending_transfer
        st.session_state.pending_transfer = None  # clear immediately

        target_dept = transfer["target_dept"]
        target_name = transfer["target_name"]
        original_query = transfer["query"]

        # Show a system message about the transfer
        transfer_msg = f" **Transferring you to {target_name}...**"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": transfer_msg,
        })
        with st.chat_message("assistant", avatar="\U0001f916"):
            st.markdown(transfer_msg)

        # Re-run the query through the target department
        with st.chat_message("assistant", avatar="\U0001f916"):
            with st.spinner(f"Connecting to {target_name}..."):
                from agents import run_query_for_department
                agent = load_agent()
                ctx = build_user_context(user)
                result = run_query_for_department(
                    query=original_query,
                    target_department=target_dept,
                    user_id=user["email"],
                    agent=agent,
                    user_context=ctx,
                )

            response = result.get("response", "Sorry, something went wrong.")
            st.markdown(response)

            meta = {
                "department": result.get("department", ""),
                "sentiment": result.get("sentiment", ""),
                "severity": result.get("severity", ""),
                "quality_score": result.get("quality_score", 0),
                "escalated": bool(result.get("escalation_info")),
                "crew_report": result.get("escalation_info", {}).get("crew_report"),
                "suggested_transfer": result.get("suggested_transfer", ""),
            }

            cols = st.columns(5)
            cols[0].caption(f"Dept: {meta['department']}")
            cols[1].caption(f"Sentiment: {meta['sentiment']}")
            cols[2].caption(f"Severity: {meta['severity']}")
            cols[3].caption(f"Quality: {meta['quality_score']}/10")
            cols[4].caption(f"Escalated: {'Yes' if meta['escalated'] else 'No'}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "meta": meta,
        })

        followup = "Is there anything else I can help you with?"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": followup,
        })

    # -- Chat input ------------------------------------------------
    if prompt := st.chat_input("Type your message..."):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="\U0001f9d1"):
            st.markdown(prompt)

        # Detect end-of-conversation responses so we don't route them
        # through the full agent pipeline (which would escalate them)
        _farewell_phrases = [
            "no thanks", "no thank you", "nope", "no thx", "that's all",
            "thats all", "nothing else", "all good", "i'm good", "im good",
            "bye", "goodbye", "good bye", "see you", "take care", "thanks bye",
            "no", "nah", "not right now", "maybe later", "that will be all",
            "nothing more", "all done", "i'm done", "im done", "bye now",
            "no thanks bye", "cheers", "thanks that's all", "thanks thats all",
        ]
        # Normalise: lowercase, strip punctuation
        _prompt_lower = prompt.strip().lower()
        _prompt_clean = "".join(ch for ch in _prompt_lower if ch.isalnum() or ch == " ").strip()

        # Match if the WHOLE input is a farewell phrase, OR if a short
        # message (<= 8 words) contains at least one farewell phrase.
        _is_farewell = _prompt_clean in [p.replace("'", "") for p in _farewell_phrases]
        if not _is_farewell and len(_prompt_clean.split()) <= 8:
            _is_farewell = any(fp.replace("'", "") in _prompt_clean for fp in _farewell_phrases)

        if _is_farewell:
            # Friendly sign-off  --  no agent call needed
            farewell_response = (
                f"Thank you for chatting with us, **{user['name']}**! "
                "If you ever need help again, just come back and start a new chat. "
                "Have a great day! \U0001f60a"
            )
            with st.chat_message("assistant", avatar="\U0001f916"):
                st.markdown(farewell_response)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": farewell_response,
            })
            st.rerun()
            st.stop()

        # Get agent response
        with st.chat_message("assistant", avatar="\U0001f916"):
            with st.spinner("Thinking..."):
                agent = load_agent()
                ctx = build_user_context(user)
                result = query_agent(agent, prompt, user_id=user["email"], user_context=ctx)

            response = result.get("response", "Sorry, something went wrong.")
            st.markdown(response)

            # Metadata row
            meta = {
                "department": result.get("department", ""),
                "sentiment": result.get("sentiment", ""),
                "severity": result.get("severity", ""),
                "quality_score": result.get("quality_score", 0),
                "escalated": bool(result.get("escalation_info")),
                "crew_report": result.get("escalation_info", {}).get("crew_report"),
                "suggested_transfer": result.get("suggested_transfer", ""),
            }

            cols = st.columns(5)
            cols[0].caption(f"Dept: {meta['department']}")
            cols[1].caption(f"Sentiment: {meta['sentiment']}")
            cols[2].caption(f"Severity: {meta['severity']}")
            cols[3].caption(f"Quality: {meta['quality_score']}/10")
            cols[4].caption(f"Escalated: {'Yes' if meta['escalated'] else 'No'}")

            # CrewAI report expandable
            if meta.get("crew_report"):
                with st.expander("View CrewAI Investigation Report"):
                    crew = meta["crew_report"]
                    for section in ("analysis", "investigation", "resolution"):
                        if crew.get(section):
                            st.markdown(f"**{section.upper()}**")
                            st.write(crew[section][:800])

            # Escalation reference
            esc_info = result.get("escalation_info", {})
            if esc_info.get("reference_number"):
                st.info(f"Escalation Reference: **{esc_info['reference_number']}**")

            # Cross-department handoff suggestion
            suggested = result.get("suggested_transfer", "")
            if suggested:
                from config import DEPARTMENTS as _DEPTS
                target_name = _DEPTS.get(suggested, {}).get("name", suggested)
                st.warning(
                    f"The {meta['department'].replace('_', ' ').title()} Agent "
                    f"thinks your query is better handled by **{target_name}**."
                )
                if st.button(
                    f"\U0001f500 Yes, transfer me to {target_name}",
                    key=f"transfer_{len(st.session_state.chat_history)}",
                    use_container_width=True,
                ):
                    st.session_state.pending_transfer = {
                        "query": prompt,
                        "target_dept": suggested,
                        "target_name": target_name,
                    }
                    st.rerun()

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "meta": meta,
        })

        # Follow-up prompt
        followup = "Is there anything else I can help you with?"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": followup,
        })

    # -- Follow-up message (always show after any conversation) ----
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last["role"] == "assistant" and last["content"].startswith("Is there anything"):
            with st.chat_message("assistant", avatar="\U0001f916"):
                st.markdown(last["content"])

    # -- Switch User button (prominent, in main area) -------------
    if st.session_state.chat_history:
        st.markdown("")
        col_a, col_b, col_c = st.columns([2, 1, 2])
        with col_b:
            if st.button("\U0001f504 Switch User", use_container_width=True, key="switch_user"):
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.chat_history = []
                st.rerun()


# =====================================================================
#  MAIN
# =====================================================================
if st.session_state.authenticated:
    show_chat()
else:
    show_login()
