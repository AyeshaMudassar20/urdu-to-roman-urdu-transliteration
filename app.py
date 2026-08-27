"""
Streamlit demo for the Urdu -> Roman Urdu Seq2Seq transliteration model.

Run with:
    streamlit run app.py

Requires a trained checkpoint at best_seq2seq_model.pkl in the repo root
(not included in this repo -- see README for how to obtain or retrain it).
"""

import streamlit as st
import torch

from dataset import load_tokenizers
from model import build_model
from transliterate import transliterate

CHECKPOINT_PATH = "best_seq2seq_model.pkl"


@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ur_sp, en_sp = load_tokenizers()
    model = build_model(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ur_sp, en_sp, device


st.set_page_config(page_title="Urdu -> Roman Urdu Transliteration", layout="centered")
st.title("Urdu to Roman Urdu Transliteration")
st.write(
    "Type Urdu-script text below and get its Roman Urdu transliteration from a "
    "bidirectional LSTM encoder / 4-layer LSTM decoder Seq2Seq model."
)

try:
    model, ur_sp, en_sp, device = load_everything()
except FileNotFoundError:
    st.error(
        f"Could not find `{CHECKPOINT_PATH}`. Train the model with `python train.py` "
        "first, or place a trained checkpoint in the repo root."
    )
    st.stop()

urdu_text = st.text_area("Urdu text:", height=120, placeholder="یہاں اردو میں لکھیں...")
clicked = st.button("Transliterate")

if clicked and urdu_text.strip():
    with st.spinner("Transliterating..."):
        result = transliterate(model, ur_sp, en_sp, urdu_text.strip(), device)
    st.subheader("Roman Urdu")
    st.write(result)
elif clicked:
    st.warning("Please enter some Urdu text first.")
