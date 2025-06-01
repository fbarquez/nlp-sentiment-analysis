# IMDb Movie Review Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" />
  <img src="https://img.shields.io/badge/scikit--learn-0.24+-yellow" />
  <img src="https://img.shields.io/badge/nlp-text--processing-orange" />
  <img src="https://img.shields.io/badge/streamlit-app-success" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)

[![GitHub Repo](https://img.shields.io/badge/Repository-GitHub-blue?logo=github)](https://github.com/fbarquez/nlp-sentiment-analysis)


## Project Overview

This repository demonstrates a complete machine learning workflow for building a **binary sentiment analysis classifier** using **Natural Language Processing (NLP)** techniques on IMDb movie reviews. It includes data cleaning, feature extraction, model training, evaluation, and deployment via a Streamlit web app.

---

## What is Sentiment Analysis?

**Sentiment analysis** is a classification task in NLP that involves identifying the **emotional tone** behind words. It's often used to determine whether a piece of text (e.g., a product review or tweet) is **positive**, **negative**, or **neutral**.

In this case, we simplify the problem to a **binary classification** task:

* `positive`
* `negative`

---

## Step-by-Step Guide

This section explains what each part of the pipeline does and why it's necessary.

### 1. Data Ingestion

**File**: `IMDB Dataset.csv`

* Contains two columns:

  * `review`: The movie review (text).
  * `sentiment`: The label, either `'positive'` or `'negative'`.

We begin by loading this CSV into a Pandas DataFrame:

```python
import pandas as pd
df = pd.read_csv('data/IMDB Dataset.csv')
df.head()
```

### 2. Text Preprocessing

**File**: `notebooks/01_preprocessing.ipynb`

#### Why preprocess?

Raw text often contains unnecessary elements like HTML tags, punctuation, or stopwords. These don't contribute meaningful information and can reduce model accuracy.

#### Steps:

1. **Lowercasing**: Makes the text uniform.
2. **Removing HTML tags**: With regex `re.sub(r"<.*?>", "", text)`.
3. **Removing punctuation/digits**: Strips irrelevant characters.
4. **Tokenization**: Splits text into individual words.
5. **Stopword Removal**: Removes common words (e.g., "the", "is") using `nltk.corpus.stopwords`.
6. **Stemming**: Reduces words to their root form using `PorterStemmer` (e.g., "loved" → "love").

```python
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
```

After cleaning, we create a new column `clean_review` and export the data to a new CSV.

### 3. Feature Extraction with TF-IDF

**File**: `notebooks/02_model_training.ipynb`

#### What is TF-IDF?

**TF-IDF (Term Frequency-Inverse Document Frequency)** transforms text into a numerical matrix that reflects the importance of each word in a document relative to the whole corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
```

* `X` becomes the input features (sparse matrix).
* `y` is created from:

```python
y = df['sentiment'].map({'positive': 1, 'negative': 0})
```

### 4. Model Training with Naive Bayes

#### Why Naive Bayes?

Naive Bayes works well for text because it assumes that all words contribute independently to the label. It's fast and effective for high-dimensional data.

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
```

#### Train/Test Split

We divide the data using:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Evaluation Metrics

After training, we evaluate the model with:

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

* **Accuracy**: Overall correctness.
* **Precision**: How many selected items are relevant.
* **Recall**: How many relevant items are selected.
* **F1 Score**: Harmonic mean of precision and recall.

Example output:

```
              precision    recall  f1-score   support

           0       0.84      0.86      0.85      5000
           1       0.86      0.84      0.85      5000
```

### 6. Model Serialization

We save the model and vectorizer for reuse without retraining:

```python
import joblib
joblib.dump(model, 'models/nb_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
```

---

## Web App with Streamlit

**File**: `app/app.py`

Streamlit allows quick deployment of machine learning models with a graphical interface.

#### Features:

* Input a custom movie review.
* Click to analyze sentiment.
* View prediction + confidence level.

#### Example logic:

```python
text = preprocess(input_text)
vector = vectorizer.transform([text])
pred = model.predict(vector)
```

---

## Project Structure

```
.
├── app/                    # Streamlit app interface
├── data/                   # Raw and processed datasets
├── models/                 # Saved model + vectorizer
├── notebooks/              # EDA and model training
├── requirements.txt        # Package dependencies
└── README.md               # Project guide
```

---

## Setup & Installation

### Step-by-step:

```bash
git clone git@github.com:fbarquez/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the app locally:

```bash
cd app
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push repo to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Set repo: `fbarquez/nlp-sentiment-analysis`, file: `app/app.py`
5. Click **Deploy**

---

## Key Terms Explained

* **Stopwords**: Common words like "the" or "and" that add little value in text classification.
* **Stemming**: Cutting words down to their root ("running" → "run").
* **TF-IDF**: Measures how important a word is in a document relative to the corpus.
* **Naive Bayes**: A probabilistic classifier based on Bayes' theorem assuming feature independence.
* **Serialization**: Saving model objects to disk for reuse.

---

## Next Steps / Improvements

* Swap stemmer for a lemmatizer.
* Try deep learning (e.g. LSTM or BERT).
* Deploy on Hugging Face Spaces.
* Improve UI/UX with text visualization.

---

## Author

**@fbarquez** — Machine Learning enthusiast and NLP practitioner. Connect or fork to improve the project!

