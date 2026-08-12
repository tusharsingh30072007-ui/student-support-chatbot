import streamlit as st
import json
import pickle
import random


# -----------------------------
# Load chatbot files
# -----------------------------

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# -----------------------------
# Chatbot function
# -----------------------------

def get_response(user_input):
    user_input_vector = vectorizer.transform([user_input])

    predicted_intent = model.predict(user_input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_intent:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't understand your question."


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Student Support AI",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 17px;
    margin-bottom: 25px;
}

.service-box {
    padding: 12px;
    border-radius: 10px;
    background-color: #f1f5f9;
    margin-bottom: 8px;
}

.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🎓 Student Support")

    st.write("### Services")

    st.markdown("""
    📅 **Examination Support**

    📝 **Assignments**

    📊 **Attendance**

    💰 **Fee Information**

    📚 **Library Services**

    🎓 **Scholarships**

    🏆 **Results**

    🕐 **Class Timetable**

    🏫 **Campus Facilities**
    """)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="title">🎓 Student Support AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your intelligent assistant for student services</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Welcome message
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:

    st.info(
        "👋 Welcome! Ask me about attendance, exams, fees, "
        "scholarships, library, timetable and other student services."
    )

    st.write("### 💡 Try asking")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="service-box">📊 How can I check my attendance?</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="service-box">📅 When are the exams?</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="service-box">🎓 How can I apply for a scholarship?</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="service-box">📚 What are the library timings?</div>',
            unsafe_allow_html=True
        )


# -----------------------------
# Display chat history
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat input
# -----------------------------

user_input = st.chat_input(
    "Ask your question..."
)


if user_input:

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Bot response
    response = get_response(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    '<div class="footer">🤖 Powered by Machine Learning • TF-IDF • Logistic Regression</div>',
    unsafe_allow_html=True
)