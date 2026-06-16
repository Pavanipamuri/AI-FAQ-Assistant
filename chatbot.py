import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download punkt tokenizer
nltk.download('punkt')

# Load FAQ data
with open('faq_data.json', 'r') as file:
    faq_data = json.load(file)

questions = list(faq_data.keys())
answers = list(faq_data.values())

# Convert text to vectors
vectorizer = CountVectorizer().fit_transform(questions)
vectors = vectorizer.toarray()

print("FAQ Chatbot")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    # Transform user question
    user_vector = CountVectorizer().fit(questions + [user_input]).transform(questions + [user_input]).toarray()

    similarity = cosine_similarity([user_vector[-1]], user_vector[:-1])

    index = similarity.argmax()

    response = answers[index]

    print("Chatbot:", response)