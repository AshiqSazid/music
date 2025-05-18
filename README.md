
# 🎵 Music Intelligence Assistant and Virality track prediction
## Full report on this : https://docs.google.com/document/d/1ajDZv-EsEBVNkJ76rp-ZMhPLmOnPuq03gs1MmVOKpP0/edit?usp=sharing
##Task 1+2 Colab : https://colab.research.google.com/drive/1bTWqKOauz5yyI1CvKuWktY-8mvPiMifd?usp=sharing
An all-in-one AI-powered music analysis platform that turns audio and lyrics into deep, actionable insights. Built for musicians, data scientists, content creators, and curious minds.


---

## 🚀 What This Does

🎼 **Instrument Detection**  
→ Identifies instruments in real-time using **YAMNet** and returns **normalized proportions**.

🎵 **Mood & Genre Classification**  
→ Multimodal embeddings via **OpenL3** + **Transformers** for smart emotion and genre prediction.

📈 **Virality Prediction**  
→ Predicts how viral your track could be using a **fine-tuned Random Forest model** on audio + metadata features.

🧠 **LLM-Powered Music Insights**  
→ Uses **Mistral/DeepSeek** via **Ollama** to generate human-like interpretations of your music’s structure, emotion, and trends.

📄 **PDF + CSV Export**  
→ Generates printable reports with mood, genre, instrument maps, and LLM insights.

🎧 **Supports Audio Uploads + YouTube URLs**  
→ MP3, WAV, or drop in a YouTube link for automatic lyric extraction and analysis.

🖼️ **t-SNE Visualization**  
→ Visual clustering of tracks based on mood/genre embeddings.

🗣️ **Zero-Shot Mood/Genre Tagging**  
→ Uses Sentence-BERT to identify vibes with no prior training on your data.

🎵 **Spotify Integration**  
→ Fetches popularity score, genre tags, and artist metadata to enhance virality prediction.

🎵 **Used RandomForestRegressor and RAG for better output**
---

## 🗂️ File Structure

```

z.py                  # Main Gradio app
gardio_langchain.py   #also main gradio app using langchain
/analysis/            # Instrument, lyric, and audio analysis modules
/models/              # Pretrained + custom-trained models
/utils/               # Spotify API, visualization, feature extractors
/assets/              # Logos, icons, sample reports
README.md
requirements.txt

````

---

## 🛠️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/AshiqSazid/music.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
 z.py or,
gradio_langchain.py
````

---

## 📊 Outputs You’ll Get

* 🎹 Instrument breakdown charts
* 🎨 Mood/genre visualizations (t-SNE)
* 🧾 PDF/CSV summary reports
* 🤖 Smart feedback from a local LLM
* 🔥 Virality prediction with confidence scores

---

## 🔮 Future Upgrades

* 🎙️ Real-time performance feedback (timing, pitch, dynamics)
* 🌐 Multilingual lyric analysis + translation
* 🧬 Listener persona modeling
* 🎥 AI-generated sync visuals for music videos
* 📱 TikTok/Instagram virality score prediction
* 🤖 Auto hashtag + caption generation for musicians

---

## 💡 Tech Stack

| Layer         | Tools Used                                            |
| ------------- | ----------------------------------------------------- |
| Audio         | `librosa`, `YAMNet`, `OpenL3`, `Spleeter`             |
| NLP/Lyrics    | `Transformers`, `Sentence-BERT`, `spaCy`              |
| ML/Prediction | `RandomForest`, `scikit-learn`, `XGBoost` (option)    |
| LLM           | `Mistral` / `DeepSeek` via `Ollama` (local inference), `langchain` |
| RAG           | `Embedding-based Retrieval` / `Metadata Filtering`,`FAISS VECTOR`, |
| UI            | `Streamlit`, `matplotlib`, `t-SNE`, `Altair`          |
| Others        | `FFmpeg`, `Spotify API`, `pandas`, `pdfkit`           |

---

## 🧪 Inspired by Research

* 🎧 [AudioGPT](https://arxiv.org/abs/2304.12966)
* 🎼 [LLark](https://arxiv.org/abs/2403.00703)
* 📈 [ERO-SHOT: Music Sentiment Analysis](https://arxiv.org/abs/2402.12392)
* 🤖 Multimodal emotion classification (OpenL3 + lyrics)

---

## 📘 Citation

If you use or adapt this project, please cite it or give a GitHub star ⭐️. Credit where it’s due!

---

## 🧑‍💻 Author

Built by **Ashiq Sazid**
🎓 Music meets AI | 🛠️Ai & backend Developer | 🔬 Researcher
Let’s connect: [LinkedIn](https://www.linkedin.com/in/ashiq-sazid/) | [GitHub](https://github.com/AshiqSazid)

---

## ⚡ Try It Locally or Deploy to Hugging Face with ngork / Gradio

```
python z.py / python gradiolangchain.py(with langchain)
```

Or package it as a Hugging Face Space or Gradio Community App for public access.

```

---

```
