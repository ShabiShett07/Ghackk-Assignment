import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv('ChatbotQuestions.csv', encoding='latin1')

lemmatizer = WordNetLemmatizer()

def prep_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['proc_questions'] = data['questions'].apply(prep_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['proc_questions'])

def get_response(user_input):
    user_input_proc = prep_text(user_input)
    user_input_vector = vectorizer.transform([user_input_proc])
    
    similarities = cosine_similarity(user_input_vector, tfidf_matrix)
    index = similarities.argmax()
    
    return data['responses'].iloc[index]

print("Welcome to the Castle Swimmer Chatbot! Ask me anything from Chapter 83 to 89 of Castle Swimmer. or else type 'exit' to say Goodbye!")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print("Chatbot:", response)
