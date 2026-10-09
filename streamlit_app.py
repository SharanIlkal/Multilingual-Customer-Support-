import streamlit as st
from src.inference.chatbot import SupportChatbot
from src.models.predict import load_model

st.set_page_config(
    page_title="Customer Support Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    /* Base application */
    .stApp {
        background: #000000;
        color: #ffffff;
    }

    /* Main content spacing: prevent title clipping */
    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 1100px !important;
        width: 100% !important;
        padding-top: 3.5rem !important;
        padding-bottom: 7rem !important;
        overflow: visible !important;
    }

    /* Main title */
    [data-testid="stMainBlockContainer"] h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        line-height: 1.4 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.75rem !important;
        padding: 0.25rem 0 !important;
        height: auto !important;
        overflow: visible !important;
        overflow-wrap: normal !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #080808;
        border-right: 1px solid #292929;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #a0a0a0 !important;
    }

    /* General headings and text */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, label {
        color: #dddddd;
    }

    /* Welcome card */
    .welcome-card {
        background: #0a0a0a;
        border: 1px solid #292929;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .welcome-card h3 {
        color: #ffffff !important;
        font-size: 1.35rem;
        margin-top: 0;
        margin-bottom: 1rem;
    }

    .welcome-card p {
        color: #d0d0d0;
        line-height: 1.8;
        margin-bottom: 0.8rem;
    }

    .welcome-card p:last-child {
        margin-bottom: 0;
    }

    /* Common requests heading */
    .section-heading {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 700;
        margin: 1.5rem 0 0.8rem;
    }

    /* Buttons: white background, black text */
    div.stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ffffff !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        min-height: 44px;
    }

    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #000000 !important;
    }

    div.stButton > button:hover {
        background: #e6e6e6 !important;
        border-color: #e6e6e6 !important;
        color: #000000 !important;
    }

    div.stButton > button:focus {
        box-shadow: 0 0 0 2px #999999 !important;
    }

    /* Sidebar information cards */
    .status-card {
        background: #111111;
        border: 1px solid #2b2b2b;
        border-radius: 11px;
        padding: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .status-label {
        color: #999999;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.45rem;
    }

    .status-value {
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    [data-testid="stSidebar"] hr {
        border-color: #292929;
        margin: 1.4rem 0;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #0b0b0b;
        border: 1px solid #292929;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatMessage"] p {
        color: #ffffff !important;
    }

    /* Fixed bottom input area */
    [data-testid="stBottom"] {
        background: #0b0b0b !important;
        border-top: 1px solid #292929 !important;
        padding: 0.75rem 0 1rem !important;
    }

    /* Center input and match the content width */
    [data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
        max-width: 1100px !important;
        width: 100% !important;
        padding: 0 1rem !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }

    /* Input wrapper */
    [data-testid="stChatInput"] {
        background: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        padding: 0.35rem !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Message textarea */
    [data-testid="stChatInput"] textarea {
        background: #111111 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        min-height: 2.5rem !important;
        padding: 0.6rem !important;
        caret-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #999999 !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border: none !important;
        box-shadow: none !important;
    }

    /* Send button */
    [data-testid="stChatInput"] button {
        background: #333333 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #555555 !important;
    }

    /* Footer */
    .app-footer {
        color: #777777;
        text-align: center;
        font-size: 0.75rem;
        margin-top: 2.5rem;
        margin-bottom: 3rem;
    }

    /* Monochrome alerts */
    [data-testid="stAlert"] {
        background: #111111 !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }

    /* Responsive layout */
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        [data-testid="stMainBlockContainer"] h1 {
            font-size: 1.75rem !important;
            line-height: 1.4 !important;
        }

        .welcome-card {
            padding: 1.1rem;
        }

        [data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
            padding: 0 0.75rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def create_chatbot():
    model, vocab, _, id_to_label = load_model()
    return SupportChatbot(
        model=model,
        vocab=vocab,
        id_to_label=id_to_label,
    )


if "chatbot" not in st.session_state:
    st.session_state.chatbot = create_chatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

def reset_chat():
    st.session_state.chatbot.reset()
    st.session_state.messages = []
    st.session_state.last_result = None

def current_value(value, fallback="Not detected"):
    return value if value else fallback

def process_message(message):
    message = message.strip()
    if not message:
        return
    st.session_state.messages.append({"role": "user", "content": message})
    try:
        result = st.session_state.chatbot.process_message(message)
        st.session_state.last_result = result
        response = result.get(
            "response",
            "I could not process your request. Please try again.",
        )

        if not response:
            response = "I could not process your request. Please try again."

    except Exception:
        response = (
            "Sorry, something went wrong while processing your request. "
            "Please try again."
        )
        st.session_state.last_result = None
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.title("Customer Support")
    st.caption("Multilingual AI assistant")
    if st.button("New conversation", use_container_width=True):
        reset_chat()
        st.rerun()
    st.divider()
    st.subheader("Assistant status")
    st.markdown(
        """
        <div class="status-card">
            <div class="status-value">Online and ready to help</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Conversation details")
    state = st.session_state.chatbot.state
    confidence = None
    if st.session_state.last_result:
        confidence = st.session_state.last_result.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_text = f"{confidence * 100:.1f}%"
    else:
        confidence_text = "None"
    status_rows = {
        "Language": current_value(state.get_context("language")),
        "Intent": current_value(state.current_intent, "None"),
        "Category": current_value(state.current_category, "None"),
        "Confidence": confidence_text,
    }

    for label, value in status_rows.items():
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">{label}</div>
                <div class="status-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    st.subheader("Supported languages")
    st.caption("English")
    st.caption("Hindi")
    st.caption("Kannada")
    st.caption("Marathi")
    st.caption("Tamil")
    st.caption("Telugu")
    st.divider()
    st.caption("27 support intents")
    st.caption("Model: BiLSTM + Attention")
st.title("Customer Support Assistant")

st.markdown(
    """
    <p style="
        color: #aaaaaa;
        font-size: 0.95rem;
        margin-top: 0.1rem;
        margin-bottom: 1.5rem;
    ">
        How can we help today? Ask about orders, refunds, payments,
        delivery, invoices, or your account.
    </p>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Welcome</h3>
            <p>
                Start a conversation in your preferred language.
                I can guide you through common customer-support requests
                and ask for additional details when needed.
            </p>
            <p>
                English, Hindi, Kannada, Marathi, Tamil and Telugu
                are supported.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Common requests</div>',
        unsafe_allow_html=True,
    )
    left_column, right_column = st.columns(2)

    with left_column:
        if st.button(
            "Track my order",
            key="request_track_order",
            use_container_width=True,
        ):
            process_message("Track my order")
            st.rerun()

        if st.button(
            "Request a refund",
            key="request_refund",
            use_container_width=True,
        ):
            process_message("I need a refund")
            st.rerun()

    with right_column:
        if st.button(
            "Payment issue",
            key="request_payment",
            use_container_width=True,
        ):
            process_message("I have a payment issue")
            st.rerun()

        if st.button(
            "Delivery problem",
            key="request_delivery",
            use_container_width=True,
        ):
            process_message("I have a delivery problem")
            st.rerun()

    st.markdown(
        """
        <div class="app-footer">
            Multilingual Customer Support · 27 Intents · 6 Languages
        </div>
        """,
        unsafe_allow_html=True,
    )

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
message = st.chat_input("Type your message...")
if message and message.strip():
    process_message(message)
    st.rerun()
