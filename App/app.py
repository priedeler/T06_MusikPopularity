import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set page title
st.set_page_config(page_title="Song Hit Predictor", page_icon="🎵")

# App Header
st.title("🎵 Song Hit Predictor")
st.markdown("""
This app predicts whether a song will become a **HIT** (Spotify popularity score > 50) based on its audio features.
Input the song's features below and choose a model to see the prediction!
""")

# Sidebar for Model Selection and Spotify API
st.sidebar.header("Configuration")

# 1. Model Selection
model_options = {
    "Ensemble (All Models)": "ensemble_all_models.joblib",
    "Random Forest": "random_forest.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Naive Bayes": "naive_bayes.joblib"
}
selected_model_name = st.sidebar.selectbox("Choose a model", list(model_options.keys()))

# 2. Spotify API Credentials (Needed for Fetching)
st.sidebar.subheader("Spotify API (Optional)")
st.sidebar.info("Get your credentials at [developer.spotify.com](https://developer.spotify.com/dashboard)")
client_id = st.sidebar.text_input("Client ID", type="password")
client_secret = st.sidebar.text_input("Client Secret", type="password")

# Initialize default slider values in session state
if 'features' not in st.session_state:
    st.session_state['features'] = {
        'acousticness': 0.26, 'danceability': 0.59, 'energy': 0.65, 
        'valence': 0.48, 'tempo': 120.0, 'loudness': -7.0, 
        'speechiness': 0.08, 'instrumentalness': 0.02
    }

# Spotify Fetching Logic
spotify_url = st.text_input("🔗 Paste Spotify Track Link", placeholder="https://open.spotify.com/track/...")

if st.button("Fetch Features from Spotify"):
    if not client_id or not client_secret:
        st.error("Please provide your Spotify Client ID and Secret in the sidebar.")
    elif not spotify_url:
        st.error("Please paste a Spotify track URL.")
    else:
        try:
            # Setup Spotify client
            auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Extract track ID and fetch features
            track_id = spotify_url.split("/")[-1].split("?")[0]
            track_info = sp.track(track_id)
            features = sp.audio_features([track_id])[0]
            
            if features:
                # Update session state
                st.session_state['features'] = {
                    'acousticness': features['acousticness'],
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'valence': features['valence'],
                    'tempo': features['tempo'],
                    'loudness': features['loudness'],
                    'speechiness': features['speechiness'],
                    'instrumentalness': features['instrumentalness']
                }
                st.success(f"Fetched: **{track_info['name']}** by **{track_info['artists'][0]['name']}**")
            else:
                st.error("Could not find audio features for this track.")
        except Exception as e:
            st.error(f"Error fetching from Spotify: {str(e)}")

# Load the selected model
@st.cache_resource
def load_model(name):
    filename = model_options[name]
    if not os.path.exists(filename):
        filename = os.path.join("App", filename)
    return joblib.load(filename)

model = load_model(selected_model_name)

# Input Features Section
st.header("🎛️ Audio Features")
col1, col2 = st.columns(2)

with col1:
    acousticness = st.slider("Acousticness", 0.0, 1.0, st.session_state['features']['acousticness'], key="s_acousticness")
    danceability = st.slider("Danceability", 0.0, 1.0, st.session_state['features']['danceability'], key="s_danceability")
    energy = st.slider("Energy", 0.0, 1.0, st.session_state['features']['energy'], key="s_energy")
    valence = st.slider("Valence", 0.0, 1.0, st.session_state['features']['valence'], key="s_valence")

with col2:
    tempo = st.slider("Tempo (BPM)", 50.0, 250.0, float(st.session_state['features']['tempo']), key="s_tempo")
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, float(st.session_state['features']['loudness']), key="s_loudness")
    speechiness = st.slider("Speechiness", 0.0, 1.0, st.session_state['features']['speechiness'], key="s_speechiness")
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, st.session_state['features']['instrumentalness'], key="s_instrumentalness")

# Create input DataFrame
input_data = pd.DataFrame([[
    acousticness, danceability, energy, valence, tempo, loudness, speechiness, instrumentalness
]], columns=['acousticness', 'danceability', 'energy', 'valence', 'tempo', 'loudness', 'speechiness', 'instrumentalness'])

# Prediction
if st.button("Predict Hit status"):
    prediction = model.predict(input_data)[0]
    
    st.divider()
    if prediction == 1:
        st.success("🎉 **It's likely a HIT!**")
    else:
        st.info("📉 **It might not be a hit.**")
        
    # Main Confidence Display
    try:
        probabilities = model.predict_proba(input_data)[0]
        st.write(f"**Overall Ensemble Confidence (Hit):** {probabilities[1]:.2%}")
    except:
        pass

    # Voting Breakdown (Specific for Ensemble)
    if selected_model_name == "Ensemble (All Models)":
        with st.expander("🔍 See Voting Breakdown", expanded=True):
            st.write("How each individual model voted:")
            display_names = {'lr': 'Logistic Regression', 'knn': 'K-Nearest Neighbors', 'rf': 'Random Forest', 'nb': 'Naive Bayes'}
            breakdown_data = []
            for name, est in model.named_estimators_.items():
                prob = est.predict_proba(input_data)[0]
                pred = "Hit" if prob[1] >= 0.5 else "Not Hit"
                breakdown_data.append({"Model": display_names.get(name, name), "Prediction": pred, "Confidence (Hit)": f"{prob[1]:.2%}"})
            st.table(pd.DataFrame(breakdown_data))
    else:
        try:
            probabilities = model.predict_proba(input_data)[0]
            st.write(f"Model Confidence (Hit): {probabilities[1]:.2%}")
        except:
            pass

st.sidebar.markdown("---")
st.sidebar.write("Developed for T06 Musik Popularity project.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for T06 Musik Popularity project.")
