import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

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

# 2. Spotify API Credentials
st.sidebar.subheader("Spotify API (Required for Fetching)")
st.sidebar.info("Get your credentials at [developer.spotify.com](https://developer.spotify.com/dashboard)")
client_id = st.sidebar.text_input("Client ID", value=os.getenv("SPOTIPY_CLIENT_ID", ""), type="password")
client_secret = st.sidebar.text_input("Client Secret", value=os.getenv("SPOTIPY_CLIENT_SECRET", ""), type="password")
redirect_uri = st.sidebar.text_input("Redirect URI", value=os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8080/callback"))

if st.sidebar.button("🗑️ Reset Spotify Login/Cache"):
    if os.path.exists(".cache"):
        os.remove(".cache")
        st.sidebar.success("Cache cleared! Try fetching again.")
    else:
        st.sidebar.info("No cache file found.")

# Initialize default slider values in session state
if 'features' not in st.session_state:
    st.session_state['features'] = {
        'acousticness': 0.26, 'danceability': 0.59, 'energy': 0.65, 
        'valence': 0.48, 'tempo': 120.0, 'loudness': -7.0, 
        'speechiness': 0.08, 'instrumentalness': 0.02
    }

# Spotify Fetching Logic
st.header("🎵 Choose a Song to Test")

# 1. New: Sample Tracks from Dataset (Ensures demo works even without API)
sample_tracks = {
    "--- Select a Sample Song ---": None,
    "Hypothetical Pop Hit (High Energy)": [0.01, 0.8, 0.9, 0.7, 128.0, -4.5, 0.05, 0.0],
    "Hypothetical Indie Song (Acoustic)": [0.85, 0.4, 0.3, 0.2, 90.0, -12.0, 0.03, 0.1],
    "Real Track: 'As It Was' Style": [0.34, 0.67, 0.73, 0.66, 174.0, -5.3, 0.05, 0.001],
}

selected_sample = st.selectbox("Quick Test: Use a sample track", list(sample_tracks.keys()))

if selected_sample != "--- Select a Sample Song ---":
    vals = sample_tracks[selected_sample]
    st.session_state['features'] = {
        'acousticness': vals[0], 'danceability': vals[1], 'energy': vals[2], 
        'valence': vals[3], 'tempo': vals[4], 'loudness': vals[5], 
        'speechiness': vals[6], 'instrumentalness': vals[7]
    }
    st.info(f"Loaded stats for: {selected_sample}")

st.markdown("---")
st.write("**OR: Fetch from Spotify (Requires API Keys)**")
spotify_url = st.text_input("🔗 Paste Spotify Track Link", placeholder="https://open.spotify.com/track/...")

if st.button("Fetch Features from Spotify"):
    if not client_id or not client_secret or not redirect_uri:
        st.error("Please provide Client ID, Secret, and Redirect URI in the sidebar to use this feature.")
    elif not spotify_url:
        st.error("Please paste a Spotify track URL.")
    else:
        try:
            # Setup Spotify OAuth
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-read-private",
                open_browser=False
            )
            sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Extract track ID
            track_id = spotify_url.split("/")[-1].split("?")[0]
            
            # Step 1: Basic track info
            track_info = sp.track(track_id)
            st.info(f"Found song: **{track_info['name']}** by **{track_info['artists'][0]['name']}**.")
            
            # Step 2: Audio features
            features_list = sp.audio_features([track_id])
            features = features_list[0] if features_list else None
            
            if features:
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
                st.success(f"Successfully fetched features for **{track_info['name']}**!")
                st.rerun()
            else:
                st.error("⚠️ **Spotify API Restriction (403)**: Spotify found the song but refused to share its audio stats. As of Nov 2024, Spotify has limited this data to 'Verified Apps' only. Please use the **Sample Song** dropdown or **Manual Sliders** for your demo.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

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
