from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import gradio as gr
import json
import pandas as pd

# data
with open("results.json", "r") as f1:
    task1_data = json.load(f1)
with open("prediction_result.json", "r") as f2:
    task2_data = json.load(f2)

popularity_df = pd.read_csv("playlist_with_popularity.csv", encoding="ISO-8859-1")

#Interpret virality
def interpret_virality(score):
    if score >= 80:
        return "🟢 Viral asf"
    elif score >= 70:
        return "🟡 Okayish viral"
    elif score >= 50:
        return "🔴 Not that viral"
    return "⚪ Flop"

#Get virality from CSV
def get_csv_virality(index):
    try:
        return int(popularity_df.iloc[index]['track_popularity'])
    except:
        return 0

#Format summary
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

#LangChain Mistral LLM
llm = Ollama(model="mistral")

prompt_template = PromptTemplate(
    input_variables=["summary", "question"],
    template="""
{summary}

User Question: {question}

🧠 Answer:
"""
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

#Main Function
def analyze_audio(index, user_question, audio_file):
    summary = format_song_summary(index)
    response = llm_chain.run(summary=summary, question=user_question)
    return summary, response

#gradio UI
demo = gr.Interface(
    fn=analyze_audio,
    inputs=[
        gr.Slider(0, len(task2_data["predictions_sample"]) - 1, step=1, label="🎵 Sample Index"),
        gr.Textbox(lines=2, label="🗣️ Ask Mistral (via LangChain)"),
        gr.Audio(label="🔊 Upload audio file (optional)", type="filepath")
    ],
    outputs=[
        gr.Textbox(label="🎼 Structured Song Summary"),
        gr.Textbox(label="🤖 LangChain Mistral's Response")
    ],
    title="🎶 LangChain Music Emotion AI",
    description="LLM + Music Intelligence using LangChain + Mistral + Gradio"
)

if __name__ == "__main__":
    demo.launch(share=False)
