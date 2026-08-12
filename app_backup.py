import streamlit as st
import json
import pickle
import random

# Load chatbot data
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Load trained model
with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


def get_response(user_input):
    user_input_vector = vectorizer.transform([user_input])

    predicted_intent = model.predict(user_input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_intent:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't understand your question."


# Page configuration
st.set_page_config(
    page_title="Student Support AI",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Support AI")
st.write("Ask me anything about student support services.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask your question...")

if user_input:
    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    response = get_response(user_input)

    # Display bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)