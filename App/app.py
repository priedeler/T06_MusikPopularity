import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page title
st.set_page_config(page_title="Song Hit Predictor", page_icon="🎵")

# App Header
st.title("🎵 Song Hit Predictor")
st.markdown("""
This app predicts whether a song will become a **HIT** (Spotify popularity score > 50) based on its audio features.
Input the song's features below and choose a model to see the prediction!
""")

# Sidebar for Model Selection
st.sidebar.header("Model Selection")
model_options = {
    "Ensemble (All Models)": "ensemble_all_models.joblib",
    "Random Forest": "random_forest.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Naive Bayes": "naive_bayes.joblib"
}
selected_model_name = st.sidebar.selectbox("Choose a model", list(model_options.keys()))

# Load the selected model
@st.cache_resource
def load_model(name):
    filename = model_options[name]
    # Check if the file exists
    if not os.path.exists(filename):
        # Maybe it's in the App folder if run from there
        filename = os.path.join("App", filename)
    return joblib.load(filename)

model = load_model(selected_model_name)

# Input Features Section
st.header("🎛️ Audio Features")
col1, col2 = st.columns(2)

with col1:
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.26, help="Confidence measure from 0.0 to 1.0 of whether the track is acoustic.")
    danceability = st.slider("Danceability", 0.0, 1.0, 0.59, help="How suitable a track is for dancing.")
    energy = st.slider("Energy", 0.0, 1.0, 0.65, help="Measure of intensity and activity.")
    valence = st.slider("Valence", 0.0, 1.0, 0.48, help="Musical positiveness (happiness, cheerfulness, etc.)")

with col2:
    tempo = st.slider("Tempo (BPM)", 50.0, 250.0, 120.0, help="The overall estimated tempo of a track in beats per minute.")
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -7.0, help="Overall loudness of a track in decibels.")
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.08, help="Presence of spoken words in a track.")
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.02, help="Predicts whether a track contains no vocals.")

# Create input DataFrame
input_data = pd.DataFrame([[
    acousticness, danceability, energy, valence, tempo, loudness, speechiness, instrumentalness
]], columns=['acousticness', 'danceability', 'energy', 'valence', 'tempo', 'loudness', 'speechiness', 'instrumentalness'])

# Prediction
if st.button("Predict Hit status"):
    prediction = model.predict(input_data)[0]
    # probability = model.predict_proba(input_data)[0][1] # Not all models have predict_proba easily (KNN/NB do, but let's check)
    
    st.divider()
    if prediction == 1:
        st.success("🎉 **It's likely a HIT!**")
    else:
        st.info("📉 **It might not be a hit.**")
        
    # Optional: show probabilities if available
    try:
        probabilities = model.predict_proba(input_data)[0]
        st.write(f"Confidence (Hit): {probabilities[1]:.2%}")
        st.write(f"Confidence (Not Hit): {probabilities[0]:.2%}")
    except:
        pass

st.sidebar.markdown("---")
st.sidebar.write("Developed for T06 Musik Popularity project.")
