"""
app.py
------
Streamlit interface for real-time inference with the AlphaSound model.

Usage
-----
    # From the project root, with the virtual environment activated:
    streamlit run app.py

Requires
--------
    pip install streamlit
    (the rest of the dependencies are already installed)
"""

import sys
import os
import tempfile
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# Add src/ to the path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_and_preprocess
from features import extract_features, normalize_spectrogram
from model import AudioLetterClassifier
from dataset import IDX_TO_LETTER, IDX_TO_LANG, ALL_LETTERS

# ── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaSound Demo",
    page_icon="🎤",
    layout="centered",
)

# ── Model loading (cached to avoid reloading on every interaction) ──────────

CHECKPOINT_PATH = "checkpoints/best_model.pt"
N_LETTERS = len(ALL_LETTERS)   # 27 (a–z + ñ)
N_LANGS = 2


@st.cache_resource
def load_model():
    """Load the trained model once and keep it in memory."""
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    model = AudioLetterClassifier(
        n_letters=N_LETTERS,
        n_langs=N_LANGS,
    )

    if not os.path.exists(CHECKPOINT_PATH):
        return None, device  # model not trained yet

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, device


# ── Inference pipeline ───────────────────────────────────────────────────────

def predict(audio_path: str, model, device) -> dict:
    """
    Full pipeline: audio → waveform → mel-spectrogram → model → result.

    Returns
    -------
    dict with:
        letter        : str   — predicted letter
        language      : str   — predicted language
        letter_conf   : float — letter confidence (0–1)
        lang_conf     : float — language confidence (0–1)
        letter_probs  : np.ndarray (N_LETTERS,) — probabilities for all letters
        mel           : np.ndarray (128, 128)   — spectrogram for visualization
    """
    # 1. Load and preprocess waveform
    waveform = load_and_preprocess(audio_path)

    # 2. Extract mel-spectrogram
    mel = extract_features(waveform)             # (1, 128, 128)
    mel_normalized = normalize_spectrogram(mel)  # (1, 128, 128)

    # 3. Convert to tensor and add batch dimension → (1, 1, 128, 128)
    tensor = torch.from_numpy(mel_normalized).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        letter_logits, lang_logits = model(tensor)

    letter_probs = torch.softmax(letter_logits, dim=-1).squeeze().cpu().numpy()
    lang_probs = torch.softmax(lang_logits, dim=-1).squeeze().cpu().numpy()

    letter_idx = int(letter_probs.argmax())
    lang_idx = int(lang_probs.argmax())

    return {
        "letter": IDX_TO_LETTER[letter_idx].upper(),
        "language": IDX_TO_LANG[lang_idx].capitalize(),
        "letter_conf": float(letter_probs[letter_idx]),
        "lang_conf": float(lang_probs[lang_idx]),
        "letter_probs": letter_probs,
        "mel": mel[0],  # (128, 128) unnormalized, for visualization
    }


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_melspectrogram(mel: np.ndarray) -> plt.Figure:
    """Real mel-spectrogram from recorded audio."""
    fig, ax = plt.subplots(figsize=(7, 3))
    img = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title("Mel-spectrogram", fontsize=12)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bands")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_top5(letter_probs: np.ndarray) -> plt.Figure:
    """Horizontal bars with the top-5 most likely letters."""
    top5_idx = np.argsort(letter_probs)[::-1][:5]
    top5_letters = [IDX_TO_LETTER[i].upper() for i in top5_idx]
    top5_probs = letter_probs[top5_idx]

    colors = ["#667eea" if i > 0 else "#764ba2" for i in range(5)]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(top5_letters[::-1], top5_probs[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 letters", fontsize=12)

    for bar, prob in zip(bars, top5_probs[::-1]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.2%}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────

st.title("🎤 AlphaSound — Demo")
st.markdown(
    "Record a letter from the alphabet and the model predicts **which letter** "
    "it is and in which **language** it was pronounced."
)

# Load model
model, device = load_model()

if model is None:
    st.warning(
        "⚠️ No trained checkpoint was found at "
        f"`{CHECKPOINT_PATH}`.\n\n"
        "Train the model first with:\n"
        "```\npython src/train.py --data_dir data/processed --n_letters 27\n```"
    )
    st.stop()

st.success(f"Model loaded · {N_LETTERS} letters · device: `{device}`")
st.divider()

# ── Two input modes ──────────────────────────────────────────────────────────
tab_mic, tab_file = st.tabs(["🎙️ Record", "📂 Upload file (.ogg / .wav)"])

temp_path = None

with tab_mic:
    audio_bytes = st.audio_input("Record a letter")
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes.read())
            temp_path = temp_file.name
        st.audio(temp_path)

with tab_file:
    uploaded_file = st.file_uploader(
        "Upload an audio file from the original dataset",
        type=["ogg", "wav"],
        help="Use a .ogg file from the dataset to check whether the model works "
             "with the original training data.",
    )
    if uploaded_file:
        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        st.audio(temp_path)

        # Show expected letter and language based on filename
        stem = uploaded_file.name.split(".")[0]  # example: "a_EN_1"
        expected_letter = stem[0].upper()
        expected_language = (
            "English" if "_EN_" in stem.upper() else
            "Spanish" if "_ES_" in stem.upper() else
            "?"
        )
        st.info(
            f"Expected from filename → **{expected_letter}** · {expected_language}"
        )

# ── Analysis (shared by both modes) ──────────────────────────────────────────
if temp_path and st.button("🚀 Analyze", use_container_width=True):
    with st.spinner("Analyzing audio..."):
        try:
            result = predict(temp_path, model, device)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            st.stop()

    # ── Typing effect ──────────────────────────────────────────────────────
    placeholder = st.empty()
    message = "Processing results..."
    displayed_text = ""

    for char in message:
        displayed_text += char
        placeholder.markdown(f"*{displayed_text}*")
        time.sleep(0.025)

    placeholder.empty()

    # ── Main result card ───────────────────────────────────────────────────
    lang_emoji = "🇬🇧" if result["language"] == "English" else "🇪🇸"

    st.markdown(
        f"""
        <div style="
            padding: 28px;
            border-radius: 16px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            text-align: center;
            margin-bottom: 24px;
            animation: fadeIn 0.8s ease-in-out;
        ">
            <h1 style="font-size: 72px; margin: 0;">{result['letter']}</h1>
            <h3 style="margin: 8px 0;">{lang_emoji} {result['language']}</h3>
            <p style="font-size: 16px; opacity: 0.9;">
                Letter confidence: <strong>{result['letter_conf']:.1%}</strong>
                &nbsp;·&nbsp;
                Language confidence: <strong>{result['lang_conf']:.1%}</strong>
            </p>
        </div>

        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_melspectrogram(result["mel"]))
    with col1:
        st.pyplot(plot_melspectrogram(result["mel"]))

    with col2:
        st.pyplot(plot_top5(result["letter_probs"]))
    with col2:
        st.pyplot(plot_top5(result["letter_probs"]))

    # Clean temporary file
    os.remove(temp_path)