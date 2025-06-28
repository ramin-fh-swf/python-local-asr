# Offline Audio Transcription as an Azure DevOps Pipeline Component
This project is a **proof of concept (PoC)** showcasing how to integrate offline speech recognition into an automated **Azure DevOps CI/CD pipeline**. It uses a locally stored version of Hugging Face’s `Wav2Vec2` speech-to-text model to perform audio transcription **entirely offline**—making it suitable for secure, reproducible, and internet-independent workflows.

**Note:** This project is initially developed and tested on Linux-based systems (e.g., Ubuntu, Debian, WSL). Compatibility with Windows or macOS may require additional adjustments.


## Activate Virtual Environment

To run the transcription pipeline, it's recommended to use a virtual Python environment. Follow these steps:

1. **Create a virtual environment:**
    ```bash
    python3 -m venv venv_linux
    ```

2. **Activate it:**
    ```bash
    source venv_linux/bin/activate
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

You can now run scripts, tests, and code quality checks safely inside the isolated environment.

## Run the Transcription Script
The main script is located in the project root. Use it to transcribe a `.wav` or `.mp3` audio file:

```bash
python3 wav2vec_audio_transcribe.py test_audio_files/common_voice_en_42693865.mp3
```

The output will be printed to the console as plain text.

## Run Tests
This project uses `pytest` for functional and integration testing.

```bash
python3 -m pytest
```

By default, this also generates a coverage report in the terminal and an HTML version under `tests/htmlcov/`.

You can customize the test behavior via the `pytest.ini` configuration file.

## Code Style & Quality Checks
To verify and enforce consistent code formatting, use `black` and `flake8`:

```bash
# Check formatting with Black
black --check wav2vec_audio_transcribe.py

# Auto-format to comply with Black
black wav2vec_audio_transcribe.py

# Check code quality and PEP8 compliance
flake8 wav2vec_audio_transcribe.py
```

## Download the Model (optional)
The pretrained Wav2Vec2 model is already included in the repository via **Git LFS** and should be available after cloning.  

If anything is missing or corrupted, you can re-download the model using the provided helper script:

```bash
python3 download_model.py
```

This script can also be customized to fetch a different Wav2Vec2 variant if desired.  
By default, the model will be saved to:

```
./models/local_wav2vec2_base/
