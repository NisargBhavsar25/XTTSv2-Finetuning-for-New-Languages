# Indic Polyglot TTS Training with XTTSv2

This guide provides comprehensive instructions for training a multilingual Indic polyglot Text-to-Speech (TTS) model based on the XTTSv2 architecture, utilizing the open-source IndicTTS dataset.

## Table of Contents

* [Installation](#installation)
* [Training Process](#training-process)
    * [Step 1: Download Pre-trained Checkpoints](#step-1-download-pre-trained-checkpoints)
    * [Step 2: Download IndicTTS Dataset](#step-2-download-indic-tts-dataset)
    * [Step 3: Extend Vocabulary for Each Language](#step-3-extend-vocabulary-for-each-language)
    * [Step 4: Train the GPT Model](#step-4-train-the-gpt-model)
* [Using Custom Datasets](#using-custom-datasets)


## Installation

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/NisargBhavsar25/XTTSv2-Indic-Polyglot.git
cd XTTSv2-Indic-Polyglot
pip install -r requirements.txt
```


## Training Process

The training is a four-step process that involves setting up the pre-trained model and dataset, preparing the vocabulary, and fine-tuning the GPT model on the Indic languages. To reproduce our exact reuslts you can directly run the `end_to_end_train.sh` file.

### Step 1: Download Pre-trained Checkpoints

Download the official XTTSv2 pre-trained model checkpoints. These will serve as the base for fine-tuning. The checkpoints will be saved in a `checkpoints/` directory[^1].

```bash
python -m scripts.preparation.download_checkpoint --output_path checkpoints/
```


### Step 2: Download IndicTTS Dataset

Download the IndicTTS dataset, which contains audio and corresponding text for multiple Indic languages. This script will create a directory named `IndicTTS-datasets/` containing the `metadata.csv` file and audio clips[^1].

```bash
python -m scripts.preparation.download_IndicTTS_dataset
```


### Step 3: Extend Vocabulary for Each Language

For each language you intend to train, you must extend the model's vocabulary and adjust its configuration. This process incorporates language-specific characters into the model.

Run the following command for **each language** in your dataset. Replace `$language_code` with the appropriate code (e.g., `hi` for Hindi, `bn` for Bengali).

```bash
python -m scripts.preparation.extend_vocab_config \
    --output_path checkpoints/ \
    --metadata_path IndicTTS-datasets/metadata.csv \
    --language $language_code \
    --extended_vocab_size 500
```

* `--output_path`: Path to the directory containing the downloaded checkpoints.
* `--metadata_path`: Path to the dataset's metadata file.
* `--language`: The language code for which to extend the vocabulary.
* `--extended_vocab_size`: The number of new tokens to add to the vocabulary.


### Step 4: Train the GPT Model

Finally, start the GPT model training process using the prepared dataset and extended vocabulary. The script `train_gpt_xtts_balanced_new.py` is designed to handle multilingual training from a single metadata file[^1].

```bash
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
```

**Key Training Parameters:**

* **`--output_path`**: Directory where the fine-tuned model checkpoints will be saved.
* **`--metadatas`**: Path to the training metadata CSV file. The script handles multilingual data by reading the language column from this file.
* **`--num_epochs`**: Total number of training epochs.
* **`--batch_size`**: Number of samples per batch.
* **`--grad_acumm`**: Gradient accumulation steps to simulate a larger batch size.
* **`--lr`**: The learning rate for the optimizer.
* **`--save_step`**: Checkpoints are saved every specified number of steps.

## Inference with Fine-Tuned Model

Once you have a fine-tuned model, you can use the `infer_tuned.py` script to synthesize speech. You will need to provide the paths to your trained model's configuration file and checkpoint, the original vocabulary file, a reference audio for the speaker's voice, the text to synthesize, and the language code.

```bash
python -m scripts.inference.infer_tuned
```

Remember to update the checkpoit path in the file before running.

## Using Custom Datasets

While this repository is configured for the IndicTTS dataset, you can train the model on your own custom dataset. To do so, you must format your data correctly.

1. **Organize Audio Files**: Place all your `.wav` audio files in a single directory (e.g., `my_dataset/wavs/`).
2. **Create `metadata.csv`**: Create a `metadata.csv` file in your dataset's root directory. This file should link the audio files to their transcriptions and specify the language. Use a pipe `|` as the separator.

The `metadata.csv` file must follow this format[^1]:

```
audio_file|text|language
wavs/audio_001.wav|This is the first sentence.|en
wavs/audio_002.wav|Ceci est la deuxième phrase.|fr
wavs/audio_003.wav|यह तीसरा वाक्य है।|hi
```

* The `audio_file` column should contain the relative path to the audio file from the metadata file's location.
* The `language` column must contain the language code corresponding to the text.

Once your custom dataset is prepared, you can update the paths in the training commands in [Step 3](#step-3-extend-vocabulary-for-each-language) and [Step 4](#step-4-train-the-gpt-model) to point to your custom `metadata.csv` file.

## Note on Advanced Training (Optional)

This repository includes scripts for fine-tuning other components of the XTTSv2 architecture, such as the D-VAE (`scripts/training/train_dvae_xtts.py`) and the GAN-based vocoder (`scripts/training/train_gan_xtts.py`)[^1].

However, **based on experience, these additional training steps are generally not necessary**. Fine-tuning the GPT model as described in [Step 4](#step-4-train-the-gpt-model) is typically sufficient to achieve high-quality results for new languages. Fine-tuning the D-VAE and GAN may not lead to significant improvements and can sometimes degrade performance[^1].