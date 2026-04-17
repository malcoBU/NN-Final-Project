# 🎙️ AlphaSound — Alphabet Letter Classifier from Audio

> Given a `.wav` recording of a spoken letter, predict **which letter** it is and **which language** (English or Spanish) it was spoken in.

<br>

## 👥 Authors

| | Name | Role |
|---|---|---|
| 🧑‍💻 | **Gonzalo** | Model architecture & training |
| 👩‍💻 | **Lucía** | Data preprocessing & augmentation |
| 🧑‍🔬 | **Marcos** | Feature extraction & evaluation |

<br>

---

## 🗂️ Project Structure

```
AlphaSound/
│
├── data/
│   ├── raw/                  # Original .wav files
│   │   ├── english/          # A–Z recordings
│   │   └── spanish/          # A–Z + CH, LL, RR, Ñ recordings
│   └── processed/            # Mel-spectrograms (numpy arrays)
│
├── src/
│   ├── dataset.py            # DataLoader & label encoding
│   ├── preprocess.py         # Audio loading, trimming, normalization
│   ├── augment.py            # Data augmentation pipeline
│   ├── features.py           # Mel-spectrogram & MFCC extraction
│   ├── model.py              # CNN + dual-head architecture
│   ├── train.py              # Training loop
│   └── evaluate.py           # Metrics, confusion matrix
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_features.ipynb      # Feature visualization
│   └── 03_results.ipynb       # Results & confusion matrices
│
├── checkpoints/               # Saved model weights (.pt)
├── requirements.txt
└── README.md
```

<br>

---

## 🧠 Model Overview

The system uses a **shared CNN backbone** with two independent classification heads — one for the letter and one for the language.

```
Input .wav
    │
    ▼
┌─────────────────────────────────┐
│     Preprocessing               │
│  • Resample to 16 kHz           │
│  • Trim silences                │
│  • Normalize amplitude          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Feature Extraction            │
│  • Mel-spectrogram (128×128)    │
│  • MFCCs (40 coefficients)      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   CNN Backbone                  │
│  Conv2d(1→32) → MaxPool         │
│  Conv2d(32→64) → MaxPool        │
│  Conv2d(64→128) → AvgPool(4×4)  │
│  Flatten → [2048]               │
└──────────────┬──────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  ┌─────────┐     ┌─────────┐
  │  Letter │     │Language │
  │  Head   │     │  Head   │
  │Softmax  │     │Softmax  │
  │ A…Z/Ñ  │     │ EN / ES │
  └─────────┘     └─────────┘
```

**Loss function:**
```
Total Loss = 0.7 × CrossEntropy(letter) + 0.3 × CrossEntropy(language)
```

<br>

---

## 🔊 Supported Labels

| Language | Letters | Total classes |
|---|---|---|
| 🇬🇧 English | A – Z | 26 |
| 🇪🇸 Spanish | A – Z + CH, LL, RR, Ñ | 30 |

<br>

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/alphasound.git
cd alphasound

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0
torchaudio>=2.0
librosa>=0.10
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

<br>

---

## 🚀 Quick Start

### Preprocess the dataset
```python
from src.preprocess import load_and_preprocess
from src.features import extract_features

y = load_and_preprocess("data/raw/english/b/speaker1_b_01.wav")
mel = extract_features(y)   # shape: (1, 128, 128)
```

### Train the model
```bash
python src/train.py \
  --data_dir data/processed \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3
```

### Run inference on a new audio file
```python
from src.model import AudioLetterClassifier
import torch, librosa

model = AudioLetterClassifier(n_letters=30, n_langs=2)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

y = load_and_preprocess("my_recording.wav")
mel = torch.tensor(extract_features(y)).unsqueeze(0)  # (1, 1, 128, 128)

with torch.no_grad():
    letter_logits, lang_logits = model(mel)
    letter = letter_logits.argmax().item()
    lang   = lang_logits.argmax().item()

print(f"Letter: {letter}  |  Language: {'English' if lang == 0 else 'Spanish'}")
```

<br>

---

## 📊 Data Augmentation

Four techniques are applied randomly during training to improve generalization:

| Technique | What it simulates | Parameter |
|---|---|---|
| **Gaussian noise** | Cheap microphones, noisy rooms | σ = 0.005 |
| **Pitch shifting** | Different voice types (deep, high) | ±2 semitones |
| **Time stretching** | Fast or slow speakers | rate ∈ [0.85, 1.15] |
| **Volume change** | Far/close microphone distance | gain ∈ [0.70, 1.30] |

<br>

---

## 📈 Evaluation

After training, run:
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

This will output:
- Overall **accuracy** for letter and language prediction
- **F1-score per class** (useful for spotting hard letters like B/V in Spanish)
- **Confusion matrix** for both heads
- Export to **ONNX** for deployment

<br>

---

## 💡 Design Decisions

- **Why Mel-spectrogram?** It mirrors how the human ear perceives sound — higher resolution at low frequencies, lower at high. This makes tonal differences between phonemes visually separable for the CNN.
- **Why a shared backbone?** Letter identity and language share many acoustic features (formants, voice onset time). Training them jointly forces the model to learn richer representations.
- **Why weighted loss (0.7 / 0.3)?** Letter prediction is harder (30 classes vs 2) and is the primary task, so it gets a larger gradient contribution.

<br>

---

## 🗺️ Roadmap

- [x] Preprocessing & augmentation pipeline
- [x] Mel-spectrogram feature extraction
- [x] CNN + dual-head architecture
- [x] Training loop with combined loss
- [ ] Wav2Vec 2.0 fine-tuning for low-data scenarios
- [ ] Real-time inference demo (microphone input)
- [ ] REST API with FastAPI
- [ ] Mobile export via TFLite

<br>

---

## 📄 License

This project is licensed under the **MIT License** — see `LICENSE` for details.
