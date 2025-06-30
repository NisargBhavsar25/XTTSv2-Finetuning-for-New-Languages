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
    
    def get_batch_samples(self, batch_idx=0, extended_size=None):
        """Get samples for a specific batch to inspect batch formation."""
        batch_size_to_use = extended_size if extended_size else self.batch_size
        
        if len(self.language_groups) <= 1:
            start_idx = batch_idx * batch_size_to_use
            end_idx = start_idx + batch_size_to_use
            return self.samples[start_idx:end_idx]
        
        # For multilingual case, create one batch
        batch_samples = []
        samples_per_lang = max(1, batch_size_to_use // len(self.languages))
        
        # Add samples from each language to this batch
        for lang in self.languages:
            start_idx = batch_idx * samples_per_lang
            end_idx = start_idx + samples_per_lang
            
            if end_idx <= len(self.language_groups[lang]):
                lang_samples = self.language_groups[lang][start_idx:end_idx]
                batch_samples.extend(lang_samples)
        
        if self.shuffle:
            random.shuffle(batch_samples)
            
        return batch_samples[:batch_size_to_use]
    
    def get_language_info(self):
        """Get information about language distribution."""
        return {
            'languages': self.languages,
            'language_counts': {lang: len(samples) for lang, samples in self.language_groups.items()},
            'total_samples': len(self.samples)
        }
    
    def get_script_analysis(self, samples):
        """Analyze the script types in the samples."""
        script_info = defaultdict(list)
        
        for sample in samples:
            text = sample.get('text', '')
            filename = os.path.basename(sample.get('audio_file', ''))
            
            # Simple script detection based on Unicode ranges
            if any('\u0d00' <= char <= '\u0d7f' for char in text):  # Malayalam
                script_info['Malayalam'].append(filename)
            elif any('\u0b80' <= char <= '\u0bff' for char in text):  # Tamil
                script_info['Tamil'].append(filename)
            elif any('\u0c80' <= char <= '\u0cff' for char in text):  # Kannada
                script_info['Kannada'].append(filename)
            elif any('\u0c00' <= char <= '\u0c7f' for char in text):  # Telugu
                script_info['Telugu'].append(filename)
            else:
                script_info['Other'].append(filename)
                
        return dict(script_info)

def create_xtts_trainer_parser():
    parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer - Batch Inspection")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to pretrained + checkpoint model")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv_path,language_identifier")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Mini batch size")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Max audio length")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Max text length")
    parser.add_argument("--batch_to_inspect", type=int, default=0,
                        help="Which batch to inspect (0-based index)")
    parser.add_argument("--extended_batch_size", type=int, default=16,
                        help="Extended batch size for more sample inspection")
    parser.add_argument("--show_multiple_batches", type=int, default=1,
                        help="Number of batches to show")

    return parser

def inspect_batch_formation(metadatas, batch_size, output_path, max_audio_length, max_text_length, 
                          batch_to_inspect=0, extended_batch_size=16, show_multiple_batches=1):
    """Inspect how batches are formed without starting training."""
    
    print("=" * 80)
    print("EXTENDED BATCH FORMATION INSPECTION")
    print("=" * 80)
    
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
            meta_file_val=None,
            language=language_id,
        )

        DATASETS_CONFIG_LIST.append(config_dataset)

    # load training samples
    print("\nLoading training samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.1,
    )

    print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples")

    # Extract languages from the loaded samples
    print("\nExtracting languages from dataset...")
    unique_languages = set()
    for sample in train_samples:
        lang = sample.get('language', 'unknown')
        unique_languages.add(lang)
    
    unique_languages = list(unique_languages)
    print(f"Found languages in dataset: {unique_languages}")

    # Create sampler for batch inspection
    train_sampler = MultilingualBalancedSampler(train_samples, batch_size, shuffle=False)
    
    # Print language distribution info
    train_info = train_sampler.get_language_info()
    print(f"\nLanguage distribution: {train_info['language_counts']}")
    print(f"Total training samples: {train_info['total_samples']}")

    # Inspect multiple batches
    for batch_num in range(show_multiple_batches):
        batch_idx = batch_to_inspect + batch_num
        
        print(f"\n" + "=" * 80)
        print(f"INSPECTING BATCH {batch_idx} (Extended Size: {extended_batch_size})")
        print("=" * 80)
        
        batch_samples = train_sampler.get_batch_samples(batch_idx, extended_batch_size)
        
        if not batch_samples:
            print(f"No samples found for batch {batch_idx}")
            continue
        
        print(f"Batch size: {len(batch_samples)}")
        
        # Analyze script distribution
        script_analysis = train_sampler.get_script_analysis(batch_samples)
        print(f"\nScript Analysis:")
        for script, files in script_analysis.items():
            print(f"  {script}: {len(files)} samples")
        
        # Group batch samples by language for inspection
        batch_by_language = defaultdict(list)
        for sample in batch_samples:
            lang = sample.get('language', 'unknown')
            batch_by_language[lang].append(sample)
        
        print(f"\nBatch composition:")
        for lang, samples in batch_by_language.items():
            print(f"  {lang}: {len(samples)} samples")
        
        print(f"\n" + "-" * 80)
        print("AUDIO FILES IN THIS BATCH:")
        print("-" * 80)
        
        for i, sample in enumerate(batch_samples):
            audio_file = sample.get('audio_file', 'Unknown')
            text = sample.get('text', 'Unknown')
            language = sample.get('language', 'unknown')
            
            # Get just the filename for cleaner output
            filename = os.path.basename(audio_file) if audio_file != 'Unknown' else 'Unknown'
            
            # Detect script for display
            script_type = "Other"
            if any('\u0d00' <= char <= '\u0d7f' for char in text):
                script_type = "Malayalam"
            elif any('\u0b80' <= char <= '\u0bff' for char in text):
                script_type = "Tamil"
            elif any('\u0c80' <= char <= '\u0cff' for char in text):
                script_type = "Kannada"
            elif any('\u0c00' <= char <= '\u0c7f' for char in text):
                script_type = "Telugu"
            
            print(f"{i+1:2d}. Language: {language:10s} | Script: {script_type:10s} | File: {filename}")
            print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print()
        
        print("-" * 80)
        print(f"BATCH {batch_idx} SUMMARY:")
        print(f"  Total files in batch: {len(batch_samples)}")
        print(f"  Languages represented: {list(batch_by_language.keys())}")
        print(f"  Samples per language: {[len(samples) for samples in batch_by_language.values()]}")
        print(f"  Script distribution: {dict(script_analysis)}")
        
        # Show file paths for easier verification
        print(f"\n" + "-" * 80)
        print("SAMPLE FILE PATHS (First 10):")
        print("-" * 80)
        for i, sample in enumerate(batch_samples[:10]):
            audio_file = sample.get('audio_file', 'Unknown')
            language = sample.get('language', 'unknown')
            print(f"{i+1:2d}. [{language}] {audio_file}")

if __name__ == "__main__":
    parser = create_xtts_trainer_parser()
    args = parser.parse_args()

    inspect_batch_formation(
        metadatas=args.metadatas,
        batch_size=args.batch_size,
        output_path=args.output_path,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        batch_to_inspect=args.batch_to_inspect,
        extended_batch_size=args.extended_batch_size,
        show_multiple_batches=args.show_multiple_batches
    )

    print("\nExtended batch inspection completed!")
