import streamlit as st

from src.models.predict import (
    load_model,
    predict,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multilingual Customer Support",
    page_icon=None,
    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #000000;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #aaaaaa;
        margin-bottom: 30px;
    }

    .result-box {
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #111111;
    }

    .intent {
        font-size: 25px;
        font-weight: 700;
    }

    .category {
        color: #aaaaaa;
        margin-top: 5px;
    }

    .confidence {
        font-size: 20px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    return load_model()


model, vocab, label_to_id, id_to_label = get_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Multilingual Customer Support'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer intent classification'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# INPUT
# ============================================================

message = st.text_area(
    "Customer Message",
    placeholder=(
        "Example: I want to track my order"
    ),
    height=140,
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "Classify Message",
    use_container_width=True,
):

    if not message.strip():

        st.warning(
            "Please enter a customer message."
        )

    else:

        predictions, _ = predict(
            message,
            model,
            vocab,
            id_to_label,
            top_k=3,
        )

        best = predictions[0]


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="intent">'
            f'{best["intent"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="category">'
            f'Category: {best["category"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: '
            f'{best["confidence"] * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )


        # ====================================================
        # TOP-3 PREDICTIONS
        # ====================================================

        st.subheader(
            "Top Predictions"
        )

        for rank, prediction in enumerate(
            predictions,
            start=1,
        ):

            col1, col2, col3 = st.columns(
                [0.1, 0.55, 0.35]
            )

            with col1:

                st.write(
                    f"**{rank}**"
                )

            with col2:

                st.write(
                    prediction["intent"]
                )

                st.caption(
                    prediction["category"]
                )

            with col3:

                st.write(
                    f'{prediction["confidence"] * 100:.2f}%'
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <div style="text-align:center;color:#666666;">
        BiLSTM + Attention |
        27 Customer Support Intents |
        6 Languages
    </div>
    """,
    unsafe_allow_html=True,
)