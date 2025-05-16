import gradio as gr
import json
import pandas as pd
import requests

# === Load data ===
with open("results.json", "r") as f1:
    task1_data = json.load(f1)
with open("prediction_result.json", "r") as f2:
    task2_data = json.load(f2)
# Load CSV with proper encoding
popularity_df = pd.read_csv("playlist_with_popularity.csv", encoding="ISO-8859-1")

# === Helper: interpret virality score ===
def interpret_virality(score):
    if score >= 80:
        return "🟢 Viral asf"
    elif score >= 70:
        return "🟡 Okayish viral"
    elif score >= 50:
        return "🔴 Not that viral"
    return "⚪ Flop"

# === Helper: get virality from CSV ===
def get_csv_virality(index):
    try:
        return int(popularity_df.iloc[index]['track_popularity'])
    except:
        return 0  # Default if not available

# === Format song feature summary ===
def format_song_summary(index):
    row = task2_data["predictions_sample"][index]
    instruments = list(task1_data.get("proportions", {}).keys())
    proportions = task1_data.get("proportions", {})
    virality_score = get_csv_virality(index)
    virality_label = interpret_virality(virality_score)

    summary = f" Detected Instruments: {', '.join(instruments)}\n"
    summary += " Estimated Proportions:\n" + '\n'.join([f"• {k}: {v:.1f}%" for k, v in proportions.items()]) + "\n\n"
    summary += f" Tempo: {row['tempo']:.3f}\n"
    summary += " MFCCs: " + ', '.join(f"{row.get(f'mfcc_{i}'):.2f}" for i in range(13)) + "\n"
    summary += " Spectral: " + ', '.join(f"{row.get(f'spectral_{i}'):.2f}" for i in range(7)) + "\n"
    summary += " Chroma: " + ', '.join(f"{row.get(f'chroma_{i}'):.2f}" for i in range(12)) + "\n"
    summary += f"\n Virality Score: {virality_score} → {virality_label}"
    return summary

# === Local Ollama Mistral LLM Query ===
def query_mistral(prompt):
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        return res.json().get("response", " No response from Ollama.")
    except Exception as e:
        return f" Failed to connect to Ollama: {e}"

# === Main analysis function ===
def analyze_audio(index, user_question, audio_file):
    summary = format_song_summary(index)
    full_prompt = summary + "\n\n Question: " + user_question + "\n🧠 Answer:"
    response = query_mistral(full_prompt)
    return summary, response

# === Gradio UI ===
demo = gr.Interface(
    fn=analyze_audio,
    inputs=[
        gr.Slider(0, len(task2_data["predictions_sample"]) - 1, step=1, label="🎵 Sample Index"),
        gr.Textbox(lines=2, label="🗣️ Ask Mistral (e.g., is this good for gym playlist?)"),
        gr.Audio(label="🔊 Upload audio file (optional, for context only)", type="filepath")
    ],
    outputs=[
        gr.Textbox(label=" Structured Summary + Virality"),
        gr.Textbox(label=" Mistral’s Suggestion")
    ],
    title="🎶 Music Intelligence Assistant (LLM-powered)",
    description="Upload audio, analyze song features and virality, and ask strategic questions to a local Mistral model."
)

if __name__ == "__main__":
    demo.launch(share=False)
