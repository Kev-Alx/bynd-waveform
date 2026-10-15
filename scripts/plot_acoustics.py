"""Figure 7 style mel/pitch panel with explicit common-duration cropping."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from beyondwaveform.io import load_audio, atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--clean", required=True)
parser.add_argument("--adversarial", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
x, y = load_audio(args.clean).numpy(), load_audio(args.adversarial).numpy()
length = min(len(x), len(y))
mels = []
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for audio, label, axis in tqdm(zip([x[:length], y[:length]], ["Clean", "Adversarial"], axes),
                                total=2, desc="Mel and pitch", unit="clips"):
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=1024, hop_length=256, n_mels=80, fmax=8000)
    db = librosa.power_to_db(mel, ref=np.max)
    mels.append(db)
    librosa.display.specshow(db, sr=16000, hop_length=256, x_axis="time", y_axis="mel", ax=axis)
    pitch, _, _ = librosa.pyin(audio, fmin=65, fmax=800, sr=16000, frame_length=1024, hop_length=256)
    pitch_axis = axis.twinx()
    pitch_axis.plot(librosa.times_like(pitch, sr=16000, hop_length=256), pitch, color="cyan", linewidth=1)
    pitch_axis.set_ylim(50, 850)
    pitch_axis.set_ylabel("F0 (Hz)")
    axis.set_title(label)
correlation = float(np.corrcoef(mels[0].ravel(), mels[1].ravel())[0, 1])
fig.suptitle(f"Log-mel correlation on common duration: {correlation:.4f}")
fig.tight_layout()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
atomic_json(str(args.output) + ".json", {"log_mel_correlation": correlation, "alignment": "crop both to common duration",
            "samples": length, "n_fft": 1024, "hop_length": 256, "n_mels": 80,
            "status": "independent visualization parameters; original Figure 7 sample unavailable"})
