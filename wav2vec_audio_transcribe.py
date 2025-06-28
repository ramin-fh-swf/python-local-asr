import os
import sys
import logging
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Disable oneDNN and suppress transformer warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Load local model
MODEL_SAMPLE_RATE = 16000
MODEL_PATH = "./models/local_wav2vec2_base"
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH, local_files_only=True)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH, local_files_only=True)


def transcribe(audio, sampling_rate, processor, model):
    audio = audio.numpy().astype(np.float32) / np.iinfo(np.int16).max
    inputs = processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids, skip_special_tokens=False)[0]


def read_wav(path):
    try:
        waveform, rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(
            orig_freq=rate, new_freq=MODEL_SAMPLE_RATE
        )
        waveform = resampler(waveform)
        return waveform.mean(dim=0).squeeze()
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"Error loading audio: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file path as argument.")
    else:
        path = sys.argv[1]
        audio = read_wav(path)
        if audio is not None:
            text = transcribe(audio, MODEL_SAMPLE_RATE, processor, model)
            print(text)
