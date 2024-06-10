
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
df = pd.read_csv("updated_dataset.csv")


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Sentence"], df["Sentiment"], test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
accuracy = classifier.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)

# Save the model
with open("model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

# Save the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)


