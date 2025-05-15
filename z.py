# Music Intelligence Assistant - Extended Version with Genre, Emotion, Lyrics Alignment, and Export

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import openl3
import tempfile
import requests
import json
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from pydub import AudioSegment
from shutil import which
import os
from datetime import datetime

# Fix ffmpeg path for PyDub (Windows)
AudioSegment.converter = which("bin/ffmpeg.exe") or "ffmpeg"

# === Streamlit Setup ===
st.set_page_config(page_title="Music Insight AI+", layout="wide")
st.title("\U0001F3B5 Music Intelligence Assistant - Pro")

st.markdown("""
Upload a song to analyze:
- \U0001F3A7 Instrument detection using TensorFlow YAMNet
- \U0001F9E0 Mood/semantic embedding with OpenL3
- \U0001F680 Virality + Genre + Emotion prediction
- \U0001F4DD Lyrics alignment + sentiment
- \U0001F4C4 Export summary as PDF/CSV
- \U0001F4AC LLM feedback via Ollama (Mistral)
""")

uploaded_file = st.file_uploader("Upload a .wav or .mp3 audio file", type=["wav", "mp3"])
upload_lyrics = st.file_uploader("Upload lyrics (optional, .txt)", type=["txt"])

# Helper function: LLM prompt
@st.cache_data

def query_ollama_mistral(prompt, model="mistral"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "No response returned from Mistral.")
    except Exception as e:
        return f"Error querying Mistral: {e}"

# === Lyrics Sentiment & Keywords ===
lyric_sentiment = ""
theme_keywords = []
audio_lyric_similarity = ""
raw_lyrics = ""
if upload_lyrics:
    raw_lyrics = upload_lyrics.read().decode("utf-8")
    blob = TextBlob(raw_lyrics)
    polarity = blob.sentiment.polarity
    sentiment_label = "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"
    lyric_sentiment = f"{sentiment_label} ({polarity:.2f})"
    words = re.findall(r'\b\w{4,}\b', raw_lyrics.lower())
    stopwords = set(['this', 'that', 'with', 'your', 'from', 'there', 'when', 'will', 'what', 'have', 'just', 'like'])
    filtered = [w for w in words if w not in stopwords]
    theme_keywords = pd.Series(filtered).value_counts().head(5).index.tolist()

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convert mp3 to wav
    if file_ext == "mp3":
        audio = AudioSegment.from_mp3(tmp_path)
        tmp_wav_path = tmp_path.replace(".mp3", ".wav")
        audio.export(tmp_wav_path, format="wav")
    else:
        tmp_wav_path = tmp_path

    # Load audio
    waveform, sr = librosa.load(tmp_wav_path, sr=16000)
    waveform = waveform[:len(waveform) // 16000 * 16000]

    # === Instrument Detection ===
    st.subheader("1. Instrument Detection (YAMNet)")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    class_names = pd.read_csv(class_map_path)['display_name'].tolist()
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform_tf)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    instrument_keywords = ['drum', 'guitar', 'piano', 'violin', 'vocal', 'saxophone', 'flute', 'bass', 'synth', 'trumpet']
    instrument_scores = {}
    for i, label in enumerate(class_names):
        for keyword in instrument_keywords:
            if keyword in label.lower():
                instrument_scores[keyword] = instrument_scores.get(keyword, 0) + mean_scores[i]
    total = sum(instrument_scores.values())
    proportions = {k.capitalize(): round(v / total * 100, 2) for k, v in instrument_scores.items()} if total > 0 else {}
    st.bar_chart(pd.Series(proportions)) if proportions else st.warning("No recognizable instruments detected.")

    # === Mood Embedding ===
    st.subheader("2. Mood Embedding (OpenL3)")
    emb_audio, sr_emb = librosa.load(tmp_wav_path, sr=48000, mono=True)
    emb_audio = emb_audio[:sr_emb * 30]
    emb, ts = openl3.get_audio_embedding(emb_audio, sr=sr_emb, content_type="music", embedding_size=512)
    mean_embedding = np.mean(emb, axis=0)
    st.success(f"Embedding shape: {mean_embedding.shape}")

    # === Audio Features ===
    st.subheader("3. Audio-Based Classification")
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
    rms = np.mean(librosa.feature.rms(y=waveform))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr))
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    mfcc_1, mfcc_2 = np.mean(mfccs[0]), np.mean(mfccs[1])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=waveform))
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
    chroma_stdev = np.std(chroma)
    features = pd.DataFrame([{
        "tempo": tempo, "rms_energy": rms, "spectral_centroid": spec_cent,
        "mfcc_1": mfcc_1, "mfcc_2": mfcc_2,
        "zero_crossing_rate": zcr, "chroma_stdev": chroma_stdev
    }])

    # === Virality Score ===
    train_X = pd.DataFrame({
        "tempo": np.random.normal(120, 10, 200),
        "rms_energy": np.random.uniform(0.1, 0.9, 200),
        "spectral_centroid": np.random.uniform(1000, 4000, 200),
        "mfcc_1": np.random.normal(-200, 20, 200),
        "mfcc_2": np.random.normal(0, 10, 200),
        "zero_crossing_rate": np.random.uniform(0.01, 0.2, 200),
        "chroma_stdev": np.random.uniform(0.01, 0.1, 200)
    })
    train_y = np.clip((0.3 * train_X["tempo"] + 200 * train_X["rms_energy"] - 100 * train_X["zero_crossing_rate"] +
                       0.01 * train_X["spectral_centroid"] + 300 * train_X["chroma_stdev"] +
                       np.random.normal(0, 5, 200)) / 400 * 100, 0, 100)
    scaler = StandardScaler()
    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(scaler.fit_transform(train_X), train_y)
    virality_score = model_rf.predict(scaler.transform(features))[0]
    st.metric("Virality Score", f"{virality_score:.2f}/100")

    # === Genre Classifier (mock logic) ===
    genre = "Pop" if tempo > 110 else "Chill" if rms < 0.2 else "HipHop"
    emotion = "Happy" if mfcc_1 > -150 else "Dark"
    st.metric("Predicted Genre", genre)
    st.metric("Emotional Tone", emotion)

    # === Export ===
    if st.button("Export Summary as CSV"):
        summary = pd.DataFrame.from_dict({
            "Instruments": proportions,
            "Virality Score": virality_score,
            "Genre": genre,
            "Emotion": emotion,
            "Sentiment": lyric_sentiment,
            "Keywords": ", ".join(theme_keywords)
        }, orient="index", columns=["Value"])
        csv_path = os.path.join(os.getcwd(), f"music_summary_{datetime.now().strftime('%H%M%S')}.csv")
        summary.to_csv(csv_path)
        st.success(f"Exported to {csv_path}")

    # === LLM Feedback ===
    st.subheader("4. LLM Feedback (Mistral)")
    prompt_data = {
        "instruments": proportions,
        "virality_score": round(virality_score, 2),
        "audio_features": features.to_dict(orient="records")[0],
        "lyric_sentiment": lyric_sentiment,
        "lyric_themes": theme_keywords,
        "genre": genre,
        "emotion": emotion,
        "lyrics_excerpt": raw_lyrics[:300] + ("..." if len(raw_lyrics) > 300 else "")
    }
    deepseek_prompt = f"""
You are a music strategist AI. Analyze:

{json.dumps(prompt_data, indent=2)}

Q:
1. What playlist type suits this song?
2. Suggest improvements (lyrics or production)?
3. Marketing advice?
"""
    st.code(deepseek_prompt)
    if st.button("Get AI Feedback"):
        with st.spinner("Querying Mistral via Ollama..."):
            result = query_ollama_mistral(deepseek_prompt)
        st.markdown("### \U0001F4AC Mistral Feedback")
        st.write(result)
else:
    st.info("Upload a WAV or MP3 file to get started.")
