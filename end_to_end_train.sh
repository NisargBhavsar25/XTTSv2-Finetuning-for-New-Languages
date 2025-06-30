#!/bin/bash

# ==============================================================================
# Full Training Script for Indic Polyglot XTTSv2 Model
# ==============================================================================
# This script automates the entire process of training a multilingual TTS model
# for several Indic languages using the IndicTTS dataset.
#
# Languages covered:
# - Hindi (hi), Gujarati (gu), Marathi (ma), Bengali (bn),
# - Tamil (ta), Telugu (tu), Malayalam (ml), Kannada (ka)
#
# Process:
# 1. Downloads the pre-trained XTTSv2 checkpoints.
# 2. Downloads the IndicTTS dataset.
# 3. Extends the model's vocabulary for each specified language.
# 4. Starts the final multilingual GPT model training.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Step 1: Download Pre-trained Checkpoints ---
echo "INFO: Step 1/4 - Downloading pre-trained XTTSv2 checkpoints..."
python -m scripts.preparation.download_checkpoint --output_path checkpoints/
echo "INFO: Pre-trained checkpoints downloaded successfully."
echo "--------------------------------------------------"

# --- Step 2: Download IndicTTS Dataset ---
echo "INFO: Step 2/4 - Downloading IndicTTS dataset..."
python -m scripts.preparation.download_IndicTTS_dataset
echo "INFO: IndicTTS dataset downloaded successfully."
echo "--------------------------------------------------"

# --- Step 3: Extend Vocabulary for Each Language ---
echo "INFO: Step 3/4 - Extending vocabulary for all target languages..."

# Define the list of language codes
# Note: These codes ('ma', 'tu', 'ka') are used as specified and might differ
# from ISO 639-1 standards. Ensure your dataset metadata matches these codes.
LANGUAGES=("hi" "gu" "ma" "bn" "ta" "tu" "ml" "ka")

for lang_code in "${LANGUAGES[@]}"; do
    echo "INFO: Extending vocabulary for language: $lang_code"
    python -m scripts.preparation.extend_vocab_config \
        --output_path=checkpoints/ \
        --metadata_path=IndicTTS-datasets/metadata.csv \
        --language="$lang_code" \
        --extended_vocab_size=500
    echo "INFO: Vocabulary for $lang_code extended."
done

echo "INFO: Vocabulary extension completed for all languages."
echo "--------------------------------------------------"


# --- Step 4: Train the Multilingual GPT Model ---
echo "INFO: Step 4/4 - Starting the multilingual GPT model training..."
echo "INFO: This process will take a significant amount of time and GPU resources."

CUDA_VISIBLE_DEVICES=0 python -m scripts.training.train_gpt_xtts_balanced_new \
    --output_path checkpoints/ \
    --metadatas IndicTTS-datasets/metadata.csv \
    --num_epochs 9 \
    --batch_size 8 \
    --grad_acumm 32 \
    --max_text_length 400 \
    --max_audio_length 330750 \
    --weight_decay 1e-2 \
    --lr 5e-5 \
    --save_step 5000

echo "=================================================="
echo "INFO: Training process finished!"
echo "=================================================="
