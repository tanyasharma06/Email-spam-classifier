import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords & punctuation

    y = [ps.stem(i) for i in y]  # Stemming

    return " ".join(y)

# Load Model & Vectorizer
tfidf = pickle.load(open('vectorizersms.pkl', 'rb'))
model = pickle.load(open('modelsms.pkl', 'rb'))

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

input_sms = st.text_area("Enter your message here:")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    if not transformed_sms.strip():  # If input is empty after processing
        st.warning("⚠️ Please enter a valid message!")
    else:
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.subheader("Prediction:")
        if result == 1:
            st.markdown("<h2 style='color:red;'>🚨 Spam</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>✅ Not Spam</h2>", unsafe_allow_html=True)
