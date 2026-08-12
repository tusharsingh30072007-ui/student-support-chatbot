import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load intents data
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


sentences = []
labels = []


# Prepare training data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])


# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sentences)


# Train the model
model = LogisticRegression(max_iter=1000)

model.fit(X, labels)


# Save the model
with open("chatbot_model.pkl", "wb") as file:
    pickle.dump(model, file)


# Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("Chatbot model trained successfully!")
print(f"Training examples: {len(sentences)}")
print(f"Intents: {len(set(labels))}")