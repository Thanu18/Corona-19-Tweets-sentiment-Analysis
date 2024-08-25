from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import re
import emoji
import string
import nltk
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt_tab')

app = Flask(__name__)

# Load the saved model and tokenizer
model = load_model('C:\Projects\Sentiment Analysis with Tansformer models\model (2).h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom text cleaning functions
def strip_emoji(text):
    return re.sub(emoji.get_emoji_regexp(), r"", text)

def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2

def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)

def lemmatize_text(text):
    st = ""
    for w in word_tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st

def preprocess_text(text):
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = strip_emoji(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = lemmatize_text(text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        sequences = tokenizer.texts_to_sequences([preprocessed_text])
        padded = pad_sequences(sequences, padding='post', maxlen=200)

        # Make prediction
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Mapping class indices to sentiment labels
        sentiment_map = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        }
        
        sentiment = sentiment_map.get(predicted_class, 'Unknown')
        return jsonify({'sentiment': sentiment})
    else:
        return jsonify({'error': 'No text provided'})

##if __name__ == '__main__':
    ##app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
