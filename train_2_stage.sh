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
python download_dataset_ta.py
python download_dataset_ml.py

echo "Starting Stage 1 training..."
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
--output_path checkpoints/ \
--metadatas datasets-ta/metadata_train.csv,datasets-ta/metadata_eval.csv,dravid \
--num_epochs 2 \
--batch_size 8 \
--grad_acumm 32 \
--max_text_length 400 \
--max_audio_length 330750 \
--weight_decay 1e-2 \
--lr 5e-6 \
--save_step 5000

echo "Starting Stage 2 training..."
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts_stage_2.py \
--output_path checkpoints/ \
--metadatas datasets-ml/metadata_train.csv,datasets-ml/metadata_eval.csv,dravid \
--num_epochs 2 \
--batch_size 8 \
--grad_acumm 32 \
--max_text_length 400 \
--max_audio_length 330750 \
--weight_decay 1e-2 \
--lr 5e-6 \
--save_step 5000

echo "Training completed successfully!"
