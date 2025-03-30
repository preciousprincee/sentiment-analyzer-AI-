import streamlit as st
import joblib
import re
import time
import nltk
import os
import urllib.request
from nltk.corpus import stopwords

# Download stopwords (ensure it's available)
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().split()
    text = [word for word in text if word not in STOP_WORDS]  # Remove stopwords
    return ' '.join(text)

# Convert Google Drive links to direct download URLs
MODEL_URL = "https://drive.google.com/uc?id=1ExZBNbnDMKIhxNhK8477LzH-ciF92Khr&export=download"
VECTORIZER_URL = "https://drive.google.com/uc?id=1Fj79e02-QNKw3MHtxY_sxFQNpmq9xBUX&export=download"

MODEL_PATH = "logistic_regression_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Function to load the model
@st.cache_resource
def load_model():
    # Download files if they don’t exist
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(VECTORIZER_PATH):
        urllib.request.urlretrieve(VECTORIZER_URL, VECTORIZER_PATH)

    # Load model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    return model, vectorizer

# Load once
model, vectorizer = load_model()

# Function to predict sentiment with progress tracking
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text])

    progress_bar = st.progress(0)  # Initialize progress bar
    for percent in range(1, 101, 20):  # Simulate stepwise progress
        time.sleep(0.2)  # Simulate processing delay
        progress_bar.progress(percent)  # Update progress
    progress_bar.empty()  # Remove progress bar when done
    
    prediction = model.predict(transformed_text)[0]
    return "Positive" if prediction == 4 else "Negative"

# Function to display a colored sentiment card
def display_sentiment_card(text, sentiment):
    color = "#28a745" if sentiment == "Positive" else "#dc3545"  # Green for positive, Red for negative
    card_html = f'''
    <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: white;">Sentiment: {sentiment}</h4>
        <p style="color: white;">{text}</p>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)

# Streamlit UI
st.title("🔍 Sentiment Analysis App")
st.write("Enter text to analyze its sentiment.")

# User input
user_input = st.text_area("Enter text:")

# Prediction button
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        display_sentiment_card(user_input, sentiment)
    else:
        st.warning("❗ Please enter some text.")
