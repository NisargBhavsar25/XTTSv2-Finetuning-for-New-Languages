import os
import gc
from collections import defaultdict
import random

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

import argparse

class MultilingualBalancedSampler:
    """Custom sampler ensuring each batch contains all languages with balanced representation."""
    
    def __init__(self, samples, batch_size, shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.language_groups = self._group_by_language()
        self.languages = list(self.language_groups.keys())
        
        if len(self.languages) == 0:
            raise ValueError("No languages found in samples")
            
        self.samples_per_lang = max(1, batch_size // len(self.languages))
        
        print(f"Found {len(self.languages)} languages: {self.languages}")
        print(f"Samples per language per batch: {self.samples_per_lang}")
        
    def _group_by_language(self):
        """Group samples by language."""
        language_groups = defaultdict(list)
        for sample in self.samples:
            language = sample.get('language', 'unknown')
            language_groups[language].append(sample)
        return dict(language_groups)
    
    def create_balanced_samples(self):
        """Create samples ensuring balanced language representation."""
        if len(self.language_groups) <= 1:
            return self.samples
            
        # Find minimum samples across all languages to ensure balanced training
        min_samples = min(len(samples) for samples in self.language_groups.values())
        samples_per_lang_total = min_samples
        
        print(f"Language sample counts: {[(lang, len(samples)) for lang, samples in self.language_groups.items()]}")
        print(f"Using {samples_per_lang_total} samples per language for balanced training")
        
        # Create balanced dataset
        balanced_samples = []
        
        # Shuffle each language group if needed
        if self.shuffle:
            for lang in self.language_groups:
                random.shuffle(self.language_groups[lang])
        
        # Create batches by cycling through languages
        batch_count = 0
        max_batches = samples_per_lang_total // self.samples_per_lang
        
        for batch_idx in range(max_batches):
            batch_samples = []
            
            # Add samples from each language to this batch
            for lang in self.languages:
                start_idx = batch_idx * self.samples_per_lang
                end_idx = start_idx + self.samples_per_lang
                
                if end_idx <= len(self.language_groups[lang]):
                    lang_samples = self.language_groups[lang][start_idx:end_idx]
                    batch_samples.extend(lang_samples)
            
            # Only add batch if it has samples from all languages
            if len(batch_samples) == len(self.languages) * self.samples_per_lang:
                if self.shuffle:
                    random.shuffle(batch_samples)
                balanced_samples.extend(batch_samples)
                batch_count += 1
        
        print(f"Created {batch_count} balanced batches with {len(balanced_samples)} total samples")
        return balanced_samples
    
    def get_language_info(self):
        """Get information about language distribution."""
        return {
            'languages': self.languages,
            'language_counts': {lang: len(samples) for lang, samples in self.language_groups.items()},
            'total_samples': len(self.samples)
        }

def create_xtts_trainer_parser():
    parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to pretrained + checkpoint model")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv_path,language_identifier")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Grad accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Max audio length")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Max text length")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Save step")

    return parser

def train_gpt(metadatas, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, max_text_length, lr, weight_decay, save_step):
    # Logging parameters
    RUN_NAME = "GPT_XTTS_FT_MULTILINGUAL"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved
    OUT_PATH = output_path

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = True  # if True it will start with evaluation
    BATCH_SIZE = batch_size
    GRAD_ACUMM_STEPS = grad_acumm

    # Define here the dataset that you want to use for the fine-tuning
    DATASETS_CONFIG_LIST = []
    for metadata in metadatas:
        train_csv, language_id = metadata.split(",")
        print(f"Loading dataset: {train_csv} with identifier: {language_id}")

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=None,  # No separate eval file
            language=language_id,  # This is just an identifier, actual languages come from CSV
        )

        DATASETS_CONFIG_LIST.append(config_dataset)

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # Use original DVAE for multilingual training
    print(f" > Using original DVAE checkpoint: {DVAE_CHECKPOINT}")

    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    # XTTS transfer learning parameters
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files([TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files([XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=11025,  # 0.5 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    
    # training parameters config
    config = GPTTrainerConfig()
    config.load_json(XTTS_CONFIG_FILE)

    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = "GPT XTTS multilingual training with balanced batches"
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.num_loader_workers = 4
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 500
    config.save_checkpoints = True
    config.print_eval = True
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [5000, 150000, 300000], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    print("Loading training samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.1,
    )

    print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples")

    # Extract languages from the loaded samples (from the CSV language column)
    print("Extracting languages from dataset...")
    unique_languages = set()
    for sample in train_samples:
        lang = sample.get('language', 'unknown')
        unique_languages.add(lang)
    
    unique_languages = list(unique_languages)
    print(f"Found languages in dataset: {unique_languages}")

    # Check if we have multiple languages for multilingual training
    if len(unique_languages) <= 1:
        print("Warning: Only one language found. Using standard training...")
        balanced_train_samples = train_samples
        balanced_eval_samples = eval_samples
    else:
        print(f"Setting up multilingual training for {len(unique_languages)} languages...")
        
        # Create balanced samples ensuring all languages in each batch
        train_sampler = MultilingualBalancedSampler(train_samples, batch_size, shuffle=True)
        eval_sampler = MultilingualBalancedSampler(eval_samples, batch_size, shuffle=False)

        # Print language distribution info
        train_info = train_sampler.get_language_info()
        print("Training language distribution:", train_info['language_counts'])

        balanced_train_samples = train_sampler.create_balanced_samples()
        balanced_eval_samples = eval_sampler.create_balanced_samples()

        print(f"Balanced training samples: {len(balanced_train_samples)}")
        print(f"Balanced evaluation samples: {len(balanced_eval_samples)}")

    # Use standard trainer with balanced samples
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=balanced_train_samples,
        eval_samples=balanced_eval_samples,
    )
    
    print("Starting training...")
    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in balanced_train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = balanced_train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, balanced_train_samples, balanced_eval_samples
    gc.collect()

    return trainer_out_path

if __name__ == "__main__":
    parser = create_xtts_trainer_parser()
    args = parser.parse_args()

    trainer_out_path = train_gpt(
        metadatas=args.metadatas,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_acumm,
        weight_decay=args.weight_decay,
        lr=args.lr,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        save_step=args.save_step
    )

    print(f"Checkpoint saved in dir: {trainer_out_path}")