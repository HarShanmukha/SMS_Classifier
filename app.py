import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# UI
st.title("📩 Text Classification App (Spam Detector)")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    st.write(f"**Prediction:** {prediction}")
