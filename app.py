from flask import Flask, request, render_template
import pickle
import re
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')  # Download stopword list (only once)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load vectorizer and models
try:
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    XGB = pickle.load(open("model_xgb.pkl", "rb"))
    BCG = pickle.load(open("model_bcg.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Error loading models/vectorizer: {e}")


port_stem = PorterStemmer()

# Text preprocessing
def clean_and_stem(text):

    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs, HTML, brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)

    # 3. Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Tokenize and remove stopwords + stem
    tokens = text.split()
    stemmed = [port_stem.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(stemmed)
    return text

# Output label mapping
def output_label(prediction):
    return "Fake News" if prediction == 0 else "Not Fake News"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news_text = request.form['news']
        cleaned_text = clean_and_stem(news_text)
        vectorized_input = vectorizer.transform([cleaned_text])

        predictions = {
            "XGradient Boosting": output_label(XGB.predict(vectorized_input)[0]),
            "BCG": output_label(BCG.predict(vectorized_input)[0])
        }

        return render_template('index.html', predictions=predictions, input_text=news_text)

    except Exception as e:
        return render_template('index.html', error=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)