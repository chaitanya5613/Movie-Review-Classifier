import numpy as np
import re                                                  # regex to strip punctuation
from tensorflow.keras.datasets import imdb                 # IMDB word -> id dictionary
from tensorflow.keras.preprocessing import sequence        # pad_sequences
from tensorflow.keras.models import load_model             # loads the trained model

MAX_LEN = 500
VOCAB_SIZE = 10000

word_index = imdb.get_word_index()
model = load_model('final_gru_sentiment_analysis.keras')

def encode(text):
    words = re.findall(r"[a-z0-9']+", text.lower())        # same cleaning as the app
    ids = []
    for w in words:
        i = word_index.get(w)                              # id, or None if not in dictionary
        ids.append(i + 3 if i is not None and i + 3 < VOCAB_SIZE else 2)   # +3 offset, 2 = unknown
    return ids

review = "An absolute masterpiece, brilliant acting and a moving story"
for n in [1, 3, 10]:                                   # same review repeated n times
    ids = encode(" ".join([review] * n))
    x = np.array([ids])                                # shape (1, length): one review, no padding at all
    score = float(model.predict(x, verbose=0)[0][0])
    print(f"repeats={n:2d} words={len(ids):3d} score={score:.3f}")