
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    model = AutoModel.from_pretrained("spoiler-shield-contrastive-model")
    tokenizer = AutoTokenizer.from_pretrained("spoiler-shield-contrastive-model")
    return model, tokenizer

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def predict_spoiler(text, anchor_embeds, model, tokenizer, threshold=0.7):
    text_emb = get_embedding(text, model, tokenizer)
    sims = torch.nn.functional.cosine_similarity(text_emb, anchor_embeds)
    return "Spoiler" if sims.max() > threshold else "Non-Spoiler"

# UI
st.title("🎬 Spoiler Shield (Prototype)")
user_input = st.text_area("Enter a movie discussion sentence:")

if user_input:
    model, tokenizer = load_model()

    # Load anchors from sample spoiler and non-spoiler examples
    anchor_df = pd.read_csv("spoiler_shield_cleaned.csv").sample(10)
    anchor_embeds = torch.stack([get_embedding(text, model, tokenizer) for text in anchor_df['text']])

    label = predict_spoiler(user_input, anchor_embeds, model, tokenizer)
    st.markdown(f"### 🔎 Prediction: **{label}**")
