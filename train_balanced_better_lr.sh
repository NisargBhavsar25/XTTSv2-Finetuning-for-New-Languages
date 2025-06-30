#!/bin/bash

# Activate the xtts environment
# source activate xtts
# OR if using conda: conda activate xtts
source ~/miniconda3/bin/activate
# If you have conda installed in a different path, adjust the above line accordingly.
conda activate xtts
# Set error handling
set -e

echo "Starting dataset downloads..."
python download_multilingual_dataset.py

echo "Dataset downloads completed successfully!"

echo "Extending vocabulary..."

python extend_vocab_config.py --output_path=checkpoints/ --metadata_path data/metadata_train.csv --language indic --extended_vocab_size 12000

echo "Starting training with balanced dataset..."

CUDA_VISIBLE_DEVICES=0 python train_gpt_balanced_new.py \
    --output_path checkpoints/ \
    --metadatas data/metadata_train.csv,indic \
    --num_epochs 7 \
    --batch_size 8 \
    --grad_acumm 32 \
    --max_text_length 400 \
    --max_audio_length 330750 \
    --weight_decay 1e-2 \
    --lr 5e-5 \
    --save_step 5000
echo "Training completed successfully!"

