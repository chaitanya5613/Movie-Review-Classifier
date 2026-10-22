# Movie Review Sentiment Analysis (BiGRU)

Classifies IMDB movie reviews as positive or negative. Built with a Bidirectional GRU in Keras and served with Streamlit.

## Results
- Test accuracy: 87.3% on the IMDB test set (25,000 reviews)
- Model: Embedding(128) -> Bidirectional GRU(64) -> Dropout(0.3) -> Dense(1, sigmoid)
- Early stopping on validation loss, best weights restored (epoch 4)

## Known limitation
The model was trained on reviews padded to 500 tokens. Short inputs scored badly when padded
(a 9-word positive review scored 0.20). Removing padding at inference raised it to 0.96.
Proper fix: retrain with mask_zero=True.

## Run it
    pip install -r requirements.txt
    streamlit run main.py

## Files
- main.py: Streamlit app (preprocessing + prediction)
- train_gru.py: training script
- final_gru_sentiment_analysis.keras: trained model
- check_len.py: experiment showing the effect of padding length on the score