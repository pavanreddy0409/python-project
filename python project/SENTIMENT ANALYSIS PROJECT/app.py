from flask import Flask, render_template, request, redirect, url_for
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the sentiment analysis model
model = load_model('sentiment_analysis.h5')

# Replace 'sentiment_labels' with the actual list of labels used during training
sentiment_labels = ["negative", "positive"]

# Function to preprocess input text
def preprocess_text(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    return tw

# Function to get sentiment prediction and probability
def get_sentiment_prediction(text):
    tw = preprocess_text(text)
    prediction = int(model.predict(tw).round().item())
    outcome = sentiment_labels[prediction].capitalize()
    prob = model.predict(tw)[0][0]
    return outcome, prob

# Route for the introduction page
@app.route('/')
def index():
    return render_template('index.html')

# Route to navigate to the review input page
@app.route('/review_input')
def review_input():
    return render_template('review_input.html')

# Route for handling the form submission and displaying the result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review']
        outcome, probability = get_sentiment_prediction(review_text)
        return render_template('result.html', review=review_text, outcome=outcome, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
