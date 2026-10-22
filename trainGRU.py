import tensorflow as tf                                   # backend
from tensorflow.keras.datasets import imdb               # IMDB dataset loader
from tensorflow.keras.preprocessing import sequence      # pad_sequences
from tensorflow.keras.models import Sequential           # linear stack of layers
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping     # stops when validation stops improving

VOCAB_SIZE = 10000   # keep only the 10,000 most frequent words
MAX_LEN = 500        # every review padded/truncated to 500 tokens

# reviews come already encoded as lists of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# pad on the left so the real review words sit at the end, same as app.py
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN, padding='pre')
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN, padding='pre')

model = Sequential([
    Embedding(VOCAB_SIZE, 128),          # turns each word id into a 128-number vector
    Bidirectional(GRU(64)),              # reads forward and backward, outputs 128 numbers
    Dropout(0.3),                        # randomly drops 30% of units to reduce overfitting
    Dense(1, activation='sigmoid'),      # one output between 0 (negative) and 1 (positive)
])

model.compile(
    optimizer='adam',                    # adaptive learning-rate optimizer
    loss='binary_crossentropy',          # loss for two-class problems
    metrics=['accuracy'],
)

early_stop = EarlyStopping(
    monitor='val_loss',                  # watch validation loss
    patience=2,                          # stop after 2 epochs with no improvement
    restore_best_weights=True,           # roll back to the best epoch
)

model.fit(
    x_train, y_train,
    validation_split=0.2,                # hold out 20% of train data for validation
    epochs=10,                           # upper limit, early stopping usually ends sooner
    batch_size=64,
    callbacks=[early_stop],
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)   # final score on 25,000 unseen reviews
print('Test accuracy:', acc)

model.save('final_gru_sentiment_analysis.keras')        # new file, keeps your old model safe