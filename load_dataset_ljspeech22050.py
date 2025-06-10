import librosa
import os
import pandas as pd
from datasets import load_dataset
import soundfile as sf
import re

LANG = "ru"
SAVE_DIR = "lj_speech_22050"
WAV_DIR = os.path.join(SAVE_DIR, "wavs")
MAX_SAMPLES = 1000

os.makedirs(WAV_DIR, exist_ok=True)

dataset = load_dataset("mozilla-foundation/common_voice_11_0", LANG, split="train")

def clean_text(text):
    text = text.replace("—", "-")
    text = text.replace("«", "\"").replace("»", "\"")
    text = text.replace("^", "")
    text = re.sub(r'["“”]', '"', text)
    return text.strip()

metadata = []

for i, item in enumerate(dataset):
    if i >= MAX_SAMPLES:
        break

    text = clean_text(item["sentence"])
    audio = item["audio"]

    y, _ = librosa.load(audio["path"], sr=22050)
    max_amp = max(abs(y))
    if max_amp > 0:
        y = y / max_amp

    filename = f"{i:05d}.wav"
    filepath = os.path.join(WAV_DIR, filename)

    sf.write(filepath, y, 22050)

    metadata.append([filename[:-4], text, text])

df = pd.DataFrame(metadata)
df.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), sep="|", header=False, index=False)
