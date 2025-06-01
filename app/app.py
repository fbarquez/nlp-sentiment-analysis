import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load('models/nb_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Text preprocessing
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="IMDb Sentiment Classifier", layout="centered")
st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.markdown("Enter a review below to detect whether the sentiment is **positive** or **negative**.")

# Example buttons
examples = {
    "Positive": "An outstanding film with a brilliant performance and emotional depth.",
    "Negative": "Terrible acting and a boring plot. I regret watching this movie."
}

st.subheader("Try an example:")
col1, col2 = st.columns(2)
if col1.button("Positive Example"):
    st.session_state['input_text'] = examples["Positive"]
if col2.button("Negative Example"):
    st.session_state['input_text'] = examples["Negative"]

# Text input
user_input = st.text_area("Your Review", value=st.session_state.get('input_text', ''))

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        clean_review = clean_text(user_input)
        vectorized = vectorizer.transform([clean_review])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0][prediction]

        label = "🟢 Positive" if prediction == 1 else "🔴 Negative"
        st.markdown(f"### Predicted Sentiment: {label}")
        st.write(f"**Confidence:** {probability * 100:.2f}%")

