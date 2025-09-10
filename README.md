# Fake-News-Detection-System
# 📰 Fake News Detection System

This project is an AI-powered system that detects whether a news article is **real** or **fake** using natural language processing and machine learning techniques. The system is designed to help readers and platforms combat misinformation by providing a reliable second opinion on the authenticity of news.

---

## 🔍 Problem Statement

With the explosive rise in digital news consumption, the spread of **fake or misleading news** has become a major global issue. These fabricated articles can manipulate public opinion, create panic, and influence elections or markets.

Traditional fact-checking methods are manual, time-consuming, and often reactive. This project addresses the need for an **automated fake news detection tool** that can instantly classify news content based on learned patterns.

---

## 🎯 Objectives

- Automatically classify news articles as **real** or **fake**.
- Use natural language processing (NLP) to clean and vectorize text.
- Train a machine learning model using labeled datasets.
- Deploy a simple and functional web application for user interaction.

---

## 🧠 Machine Learning Model

### 🔧 Preprocessing Steps:
- Removal of punctuation, digits, and special characters
- Lowercasing, stopword removal, and lemmatization
- Tokenization

### 🗂️ Feature Extraction:
- **TF-IDF (Term Frequency-Inverse Document Frequency)** Vectorizer

### 🤖 Model Used:
- **BCG (Bagging Classifier & XGBoost Classifier)**

### 🎯 Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix


## 🛠️ Tech Stack

| Technology       | Purpose                        |
|------------------|--------------------------------|
| Python           | Programming Language           |
| scikit-learn     | Machine Learning               |
| NLTK / spaCy     | Text Preprocessing             |
| Flask            | Web Framework                  |
| HTML/CSS         | Frontend UI                    |
| Pickle           | Model Serialization            |
