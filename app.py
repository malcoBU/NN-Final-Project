"""
app.py
------
Interfaz Streamlit para inferencia en tiempo real con el modelo AlphaSound.

Uso
---
    # Desde la raíz del proyecto, con el entorno virtual activado:
    streamlit run app.py

Requiere
--------
    pip install streamlit
    (el resto de dependencias ya están instaladas)
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

# Añadir src/ al path para importar los módulos del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_and_preprocess
from features   import extract_features, normalize_spectrogram
from model      import AudioLetterClassifier
from dataset    import IDX_TO_LETTER, IDX_TO_LANG, ALL_LETTERS

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaSound Demo",
    page_icon="🎤",
    layout="centered",
)

# ── Carga del modelo (cacheado para no recargar en cada interacción) ──────────

CHECKPOINT_PATH = "checkpoints/best_model.pt"
N_LETTERS       = len(ALL_LETTERS)   # 27 (a–z + ñ)
N_LANGS         = 2


@st.cache_resource
def load_model():
    """Carga el modelo entrenado una sola vez y lo mantiene en memoria."""
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    model = AudioLetterClassifier(
        n_letters=N_LETTERS,
        n_langs=N_LANGS,
    )

    if not os.path.exists(CHECKPOINT_PATH):
        return None, device  # modelo aún no entrenado

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, device


# ── Pipeline de inferencia ────────────────────────────────────────────────────

def predict(audio_path: str, model, device) -> dict:
    """
    Pipeline completo: audio → waveform → mel-spectrogram → modelo → resultado.

    Devuelve
    --------
    dict con:
        letter        : str   — letra predicha
        language      : str   — idioma predicho
        letter_conf   : float — confianza de la letra (0–1)
        lang_conf     : float — confianza del idioma (0–1)
        letter_probs  : np.ndarray (N_LETTERS,) — probabilidades de todas las letras
        mel           : np.ndarray (128, 128)   — espectrograma para visualizar
    """
    # 1. Cargar y preprocesar waveform
    y = load_and_preprocess(audio_path)

    # 2. Extraer mel-spectrogram
    mel = extract_features(y)               # (1, 128, 128)
    mel_norm = normalize_spectrogram(mel)   # (1, 128, 128)

    # 3. Convertir a tensor y añadir dimensión de batch → (1, 1, 128, 128)
    tensor = torch.from_numpy(mel_norm).unsqueeze(0).to(device)

    # 4. Inferencia
    with torch.no_grad():
        letter_logits, lang_logits = model(tensor)

    letter_probs = torch.softmax(letter_logits, dim=-1).squeeze().cpu().numpy()
    lang_probs   = torch.softmax(lang_logits,   dim=-1).squeeze().cpu().numpy()

    letter_idx = int(letter_probs.argmax())
    lang_idx   = int(lang_probs.argmax())

    return {
        "letter":       IDX_TO_LETTER[letter_idx].upper(),
        "language":     IDX_TO_LANG[lang_idx].capitalize(),
        "letter_conf":  float(letter_probs[letter_idx]),
        "lang_conf":    float(lang_probs[lang_idx]),
        "letter_probs": letter_probs,
        "mel":          mel[0],  # (128, 128) sin normalizar, para la viz
    }


# ── Visualizaciones ───────────────────────────────────────────────────────────

def plot_melspectrogram(mel: np.ndarray) -> plt.Figure:
    """Mel-spectrogram real del audio grabado."""
    fig, ax = plt.subplots(figsize=(7, 3))
    img = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title("Mel-spectrogram", fontsize=12)
    ax.set_xlabel("Frames temporales")
    ax.set_ylabel("Bandas Mel")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_top5(letter_probs: np.ndarray) -> plt.Figure:
    """Barras horizontales con el top-5 de letras más probables."""
    top5_idx  = np.argsort(letter_probs)[::-1][:5]
    top5_letters = [IDX_TO_LETTER[i].upper() for i in top5_idx]
    top5_probs   = letter_probs[top5_idx]

    colors = ["#667eea" if i > 0 else "#764ba2" for i in range(5)]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(top5_letters[::-1], top5_probs[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidad")
    ax.set_title("Top-5 letras", fontsize=12)

    for bar, prob in zip(bars, top5_probs[::-1]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.2%}",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎤 AlphaSound — Demo")
st.markdown(
    "Graba una letra del abecedario y el modelo predice **qué letra** es "
    "y en qué **idioma** fue pronunciada."
)

# Cargar modelo
model, device = load_model()

if model is None:
    st.warning(
        "⚠️ No se encontró ningún checkpoint entrenado en "
        f"`{CHECKPOINT_PATH}`.\n\n"
        "Entrena primero el modelo con:\n"
        "```\npython src/train.py --data_dir data/processed --n_letters 27\n```"
    )
    st.stop()

st.success(f"Modelo cargado · {N_LETTERS} letras · dispositivo: `{device}`")
st.divider()

# Grabación de audio
audio_bytes = st.audio_input("🎙️ Graba una letra")

if audio_bytes:
    # Guardar en fichero temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes.read())
        tmp_path = tmp.name

    st.audio(tmp_path)

    if st.button("🚀 Analizar", use_container_width=True):
        with st.spinner("Analizando audio..."):
            try:
                result = predict(tmp_path, model, device)
            except Exception as e:
                st.error(f"Error al procesar el audio: {e}")
                st.stop()

        # ── Efecto de escritura ───────────────────────────────────────────────
        placeholder = st.empty()
        msg = "Procesando resultados..."
        displayed = ""
        for char in msg:
            displayed += char
            placeholder.markdown(f"*{displayed}*")
            time.sleep(0.025)
        placeholder.empty()

        # ── Tarjeta de resultado principal ────────────────────────────────────
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
                    Confianza letra: <strong>{result['letter_conf']:.1%}</strong>
                    &nbsp;·&nbsp;
                    Confianza idioma: <strong>{result['lang_conf']:.1%}</strong>
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

        # ── Gráficas ──────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_melspectrogram(result["mel"]))

        with col2:
            st.pyplot(plot_top5(result["letter_probs"]))

        # Limpiar fichero temporal
        os.remove(tmp_path)
