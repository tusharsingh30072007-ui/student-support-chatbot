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
    # Convert user input into TF-IDF features
    user_input_vector = vectorizer.transform([user_input])

    # Predict the intent
    predicted_intent = model.predict(user_input_vector)[0]

    # Find the matching response
    for intent in data["intents"]:
        if intent["tag"] == predicted_intent:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't understand your question."


print("🎓 Student Support Chatbot")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye! Have a great day!")
        break

    response = get_response(user_input)
    print("Bot:", response)