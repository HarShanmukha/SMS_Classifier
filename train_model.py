import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("data/dataset.csv", quoting=1)  # quoting=1 => QUOTE_ALL


# ✅ Ensure correct columns
print(df.head())   # Debugging (remove later)

X = df["text"].astype(str)   # Features (messages)
y = df["label"].astype(str)  # Labels (spam/ham)

# Create pipeline
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
