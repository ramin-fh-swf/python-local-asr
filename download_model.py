from huggingface_hub import snapshot_download

snapshot_download(repo_id="facebook/wav2vec2-base-960h", local_dir="./models/local_wav2vec2_base", local_dir_use_symlinks=False)
