from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Create Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Extract opinion from form data
    opinion = request.form["Sentence"]
    # Transform the opinion into TF-IDF features
    opinion_features = tfidf_vectorizer.transform([opinion])
    # Make prediction
    prediction = model.predict(opinion_features)
    # Render prediction result in the template
    prediction_result = prediction[0]
    return render_template("index.html", prediction_text="your opinion is {}".format(prediction_result))


if __name__ == "__main__":
    app.run(debug=True)
