import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

st.title("🎭 Sentiment Analyzer")
st.write("Restaurant review ka sentiment check karo!")

user_input = st.text_area("Review likho yahan:")

if st.button("Analyze"):
    sample_vec = vectorizer.transform([user_input])
    result = model.predict(sample_vec)
    proba = model.predict_proba(sample_vec)[0]
    confidence = max(proba)

    if confidence < 0.60:
        label = "Neutral 😐"
    elif result[0] == 1:
        label = "Positive 😊"
    else:
        label = "Negative 😞"

    st.success(f"Sentiment: {label}")
    st.info(f"Confidence: {round(confidence * 100, 2)}%")