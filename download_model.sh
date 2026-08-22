#!/bin/bash
# Description: Download the Qwen3.8-27B artifacts (model + vision projector + MTP draft head).
# Usage: ./download_model.sh [qwen|gemma]      (default: qwen)
set -e

# Target directory to put models inside
MODEL_DIR="/garagedata/models/gguf"
mkdir -p "$MODEL_DIR"

WHICH="${1:-qwen}"

if [ "$WHICH" = "qwen" ]; then
    BASE="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"

    echo "Downloading Qwen3.8-27B-UD-Q4_K_XL.gguf (16.35 GiB)..."
    wget -c --show-progress -O "${MODEL_DIR}/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
        "${BASE}/Qwen3.8-27B-UD-Q4_K_XL.gguf"

    # NOTE: upstream calls this "mmproj-F16.gguf" for EVERY model. We rename it on download so it
    # cannot silently overwrite the projector of another model living in the same directory.
    echo "Downloading the vision projector (renamed to avoid clobbering other mmproj files)..."
    wget -c --show-progress -O "${MODEL_DIR}/mmproj-Qwen3.8-27B-F16.gguf" \
        "${BASE}/mmproj-F16.gguf"

    # The MTP head is shipped separately, not embedded. Needed for --spec-type draft-mtp.
    echo "Downloading the MTP draft head (speculative decoding, ~1.5x decode speed)..."
    wget -c --show-progress -O "${MODEL_DIR}/mtp-Qwen3.8-27B-Q4_0.gguf" \
        "${BASE}/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"

    FILES="Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-Qwen3.8-27B-F16.gguf mtp-Qwen3.8-27B-Q4_0.gguf"

elif [ "$WHICH" = "gemma" ]; then
    BASE="https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main"

    echo "Downloading Gemma-4-26B-A4B-it-UD-Q5_K_M.gguf..."
    wget -c --show-progress -O "${MODEL_DIR}/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf" \
        "${BASE}/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf"

    echo "Downloading mmproj-F16.gguf representing the layout projector vision component..."
    wget -c --show-progress -O "${MODEL_DIR}/mmproj-F16.gguf" \
        "${BASE}/mmproj-F16.gguf"

    FILES="gemma-4-26B-A4B-it-UD-Q5_K_M.gguf mmproj-F16.gguf"
else
    echo "Unknown target '$WHICH'. Use 'qwen' or 'gemma'." >&2
    exit 1
fi

# A truncated download or an HTML error page will not carry the GGUF magic.
echo
echo "Verifying GGUF headers..."
for f in $FILES; do
    printf "  %-40s " "$f"
    if head -c4 "${MODEL_DIR}/${f}" | grep -q GGUF; then echo "OK"; else echo "BAD - re-run to resume"; exit 1; fi
done

echo "Downloads complete!"
