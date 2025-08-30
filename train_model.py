import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin1")

# Keep only necessary columns
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Drop missing values
df = df.dropna()

# Features & Labels
X = df["text"].astype(str)
y = df["label"].astype(str)

# Build pipeline
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
