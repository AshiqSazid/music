import gradio as gr
import json
import requests

# Load local JSON files
with open("results.json", "r") as f1:
    task1_data = json.load(f1)
with open("prediction_result.json", "r") as f2:
    task2_data = json.load(f2)

llm_payload = {
    "instrument_proportions": task1_data.get("proportions", {}),
    "energy_db_duration": task1_data.get("energy_info", {}),
    "embedding_sample": {k: v[:5] for k, v in task1_data.get("embeddings", {}).items()},
    "virality_prediction_metrics": task2_data.get("metrics", {}),
    "sample_predictions": task2_data.get("predictions_sample", [])[:3]
}

context = f"""
You are an expert music analyst and marketing strategist LLM.

Here is JSON data about a song's audio breakdown, detected instruments, energy, mood embeddings, and predicted virality/popularity metrics:

{json.dumps(llm_payload, indent=2)}
"""

def chat_with_mistral(user_input):
    full_prompt = f"{context}\nUser: {user_input}\nAssistant:"
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": full_prompt,
        "stream": False
    })
    return response.json()["response"]

gr.Interface(
    fn=chat_with_mistral,
    inputs=gr.Textbox(label="Ask about the song's virality, marketing, structure..."),
    outputs=gr.Textbox(label="🎧 Mistral's Response"),
    title="🎵 Music Intelligence Assistant (LLM-Powered)",
    description="Interact with Mistral running locally via Ollama to explore your song's strengths, virality potential, and marketing ideas."
).launch()
