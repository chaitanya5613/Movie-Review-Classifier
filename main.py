import numpy as np
import re                                   # regex, used to strip punctuation
import streamlit as st                      # web UI
from tensorflow.keras.datasets import imdb  # gives us the IMDB word -> id dictionary
from tensorflow.keras.preprocessing import sequence  # pad_sequences
from tensorflow.keras.models import load_model       # loads the trained .keras model

MAX_LEN = 500        # must match maxlen used in training
VOCAB_SIZE = 10000   # must match num_words used in training


@st.cache_resource   # runs once, then reuses the result on every Streamlit rerun
def load_resources():
    word_index = imdb.get_word_index()                          # word -> integer id
    model = load_model('final_gru_sentiment_analysis.keras')   # loads the new BiGRU instead of the SimpleRNN    # trained RNN
    return word_index, model

word_index, model = load_resources()


def preprocess_text(text):
    # lowercase, keep only words (letters, digits, apostrophes); drops "!", ".", "," etc.
    words = re.findall(r"[a-z0-9']+", text.lower())

    encoded = []
    for w in words:
        idx = word_index.get(w)                      # id from the IMDB dictionary, or None
        # IMDB reserves 0=pad, 1=start, 2=unknown, so real word ids shift by +3
        if idx is not None and idx + 3 < VOCAB_SIZE:
            encoded.append(idx + 3)
        else:
            encoded.append(2)                        # unknown or out-of-vocab word

    if not encoded:                                  # input had no usable words, e.g. only "!!!"
        encoded = [2]                                # one unknown token so predict doesn't get an empty array

    return np.array([encoded])                       # shape (1, n): one review at its real length, no padding

st.title('Movie Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')            # multi-line input box

if st.button('Classify'):
    if not user_input.strip():                       # block empty or whitespace-only input
        st.warning('Please enter a movie review.')
    else:
        x = preprocess_text(user_input)
        score = float(model.predict(x, verbose=0)[0][0])   # sigmoid output between 0 and 1
        sentiment = 'Positive' if score > 0.5 else 'Negative'

        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction score: {score:.4f}')