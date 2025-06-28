import os
import torch
from wav2vec_audio_transcribe import read_wav, transcribe, MODEL_SAMPLE_RATE, processor, model

TEST_AUDIO = os.path.join(os.path.dirname(__file__), "../test_audio_files/common_voice_en_42693865.mp3")

def test_read_wav_returns_tensor():
    audio = read_wav(TEST_AUDIO)
    assert isinstance(audio, torch.Tensor)
    assert audio.dim() == 1  # Should be mono waveform
    assert not torch.isnan(audio).any()

def test_transcribe_outputs_text():
    audio = read_wav(TEST_AUDIO)
    text = transcribe(audio, MODEL_SAMPLE_RATE, processor, model)
    assert isinstance(text, str)
    assert len(text.strip()) > 0
