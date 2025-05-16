
# ======================
# 🔧 Setup Mistral + Ngrok + API
# ======================

# ✅ Install dependencies
!pip install -q transformers flask accelerate

# ✅ Download & setup ngrok (v2 compatible)
!wget -q https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip -o ngrok-stable-linux-amd64.zip

# ✅ Authenticate ngrok (replace token if needed)
!./ngrok authtoken 2xB8SqH2HExR02ztnkt03hXMKLy_27vcEy4ZVTL4V4sTzxyPK

# ✅ Load model
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

app = Flask(__name__)

@app.route("/api/generate", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": result})

# ✅ Run Flask server in background
def run_flask():
    app.run(host="0.0.0.0", port=1143)

threading.Thread(target=run_flask).start()

# ✅ Start Ngrok tunnel
import time
time.sleep(3)
!./ngrok http 1143 &

# ✅ Show public URL
time.sleep(7)
!curl -s http://localhost:4040/api/tunnels | grep -o 'https://[0-9a-z]*\.ngrok-free\.app'
