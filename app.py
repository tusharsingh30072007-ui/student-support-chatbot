import streamlit as st
import json
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
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
    # Convert user question into TF-IDF vector
    user_input_vector = vectorizer.transform([user_input])

    # Get model probabilities
    probabilities = model.predict_proba(user_input_vector)[0]

    max_probability = max(probabilities)
    predicted_intent = model.classes_[probabilities.argmax()]

    # Create TF-IDF vectors for all training patterns
    training_patterns = []
    training_tags = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            training_patterns.append(pattern)
            training_tags.append(intent["tag"])

    training_vectors = vectorizer.transform(training_patterns)

    # Calculate similarity with training examples
    similarities = cosine_similarity(
        user_input_vector,
        training_vectors
    )[0]

    max_similarity_index = similarities.argmax()
    max_similarity = similarities[max_similarity_index]

    # Check whether the question is relevant enough
    if max_similarity < 0.15:
        return (
            "Sorry, I couldn't understand your question. "
            "Please ask about student services such as attendance, "
            "exams, assignments, fees, scholarships, library, "
            "results, timetable, or campus facilities."
        )

    # Additional confidence check
    if max_probability < 0.15:
        return (
            "Sorry, I couldn't understand your question. "
            "Please ask about student services such as attendance, "
            "exams, assignments, fees, scholarships, library, "
            "results, timetable, or campus facilities."
        )

    # Return response for predicted intent
    for intent in data["intents"]:
        if intent["tag"] == predicted_intent:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't understand your question."

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

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
    color: var(--text-color);
}

.service-box {
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
}

.footer {
    text-align: center;
    font-size: 13px;
    margin-top: 30px;
    color: var(--text-color);
    opacity: 0.7;
}

[data-testid="stAlert"] {
    color: var(--text-color);
}

[data-testid="stSidebar"] {
    color: var(--text-color);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-color);
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Initialize chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🎓 Student Support")

    st.write("### Services")

    # Service buttons
    if st.button("📅 Examination Support", use_container_width=True):
        selected_query = "When are the exams?"

    elif st.button("📝 Assignments", use_container_width=True):
        selected_query = "How do I submit an assignment?"

    elif st.button("📊 Attendance", use_container_width=True):
        selected_query = "How can I check my attendance?"

    elif st.button("💰 Fee Information", use_container_width=True):
        selected_query = "How can I pay my fees?"

    elif st.button("📚 Library Services", use_container_width=True):
        selected_query = "What are the library timings?"

    elif st.button("🎓 Scholarships", use_container_width=True):
        selected_query = "How can I apply for a scholarship?"

    elif st.button("🏆 Results", use_container_width=True):
        selected_query = "Where can I check my results?"

    elif st.button("🕐 Class Timetable", use_container_width=True):
        selected_query = "Where can I find my timetable?"

    elif st.button("🏫 Campus Facilities", use_container_width=True):
        selected_query = "What facilities are available?"

    else:
        selected_query = None

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
    '<div class="subtitle">'
    'Your intelligent assistant for student services'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Welcome section
# -----------------------------

if len(st.session_state.messages) == 0:

    st.info(
        "👋 Welcome! Ask me about attendance, exams, fees, "
        "scholarships, library, timetable and other student services."
    )

    st.write("### 💡 Try asking")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="service-box">'
            '📊 How can I check my attendance?'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="service-box">'
            '📅 When are the exams?'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="service-box">'
            '🎓 How can I apply for a scholarship?'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="service-box">'
            '📚 What are the library timings?'
            '</div>',
            unsafe_allow_html=True
        )


# -----------------------------
# Process sidebar selection
# -----------------------------

if selected_query:

    response = get_response(selected_query)

    st.session_state.messages.append({
        "role": "user",
        "content": selected_query
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()


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
    '<div class="footer">'
    '🤖 Powered by Machine Learning • TF-IDF • Logistic Regression'
    '</div>',
    unsafe_allow_html=True
)