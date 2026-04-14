#!/bin/bash
# Description: Download script for the required Gemma 4 variables via curl/wget.
set -e

# Target directory to put models inside
MODEL_DIR="/data/models/gguf"
mkdir -p "$MODEL_DIR"

echo "Downloading Gemma-4-26B-A4B-it-UD-Q5_K_M.gguf..."
wget -c --show-progress -O "${MODEL_DIR}/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf" \
    "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf"

echo "Downloading mmproj-F16.gguf representing the layout projector vision component... "
wget -c --show-progress -O "${MODEL_DIR}/mmproj-F16.gguf" \
    "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-F16.gguf"

echo "Downloads complete!"
