"""
Optimized batch inference script for tuned XTTS model.
Processes a CSV file with language, generated_sentence, and sentence_id columns.
Includes dynamic batching, memory management, and performance monitoring.
Audio files are organized in language-specific subfolders.
"""

import argparse
import os
import pandas as pd
import torch
import torchaudio
import time
import asyncio
import concurrent.futures
import gc
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class OptimizedXTTSBatchInference:
    """Optimized XTTS batch inference with dynamic batching and memory management."""

    def __init__(self, config_path: str, checkpoint_path: str, vocab_path: str, device: str = "cuda"):
        self.device = device
        self.model = None
        self.config = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self.language_mapping = self._create_language_mapping()

        # Performance metrics
        self.total_processing_time = 0
        self.total_samples = 0
        self.successful_count = 0
        self.failed_count = 0

        # Load model
        self._load_model(config_path, checkpoint_path, vocab_path)

    def _create_language_mapping(self) -> Dict[str, str]:
        """Create mapping from full language names to language codes."""
        return {
            "Hindi": "hi",
            "Kannada": "ka", 
            "Malayalam": "ml",
            "Marathi": "ma",
            "Tamil": "ta",
            "Telugu": "tu",
            "Bengali": "bn",
            "Gujarati": "gu",
            "English": "en"
        }

    def _load_model(self, config_path: str, checkpoint_path: str, vocab_path: str):
        """Load the tuned XTTS model with optimization."""
        print("Loading model...")

        # Load config
        self.config = XttsConfig()
        self.config.load_json(config_path)

        # Initialize model
        self.model = Xtts.init_from_config(self.config)

        # Load checkpoint
        self.model.load_checkpoint(
            self.config, 
            checkpoint_path=checkpoint_path, 
            vocab_path=vocab_path, 
            use_deepspeed=False
        )

        self.model.to(self.device)
        self.model.eval()

        # Optimize for inference
        if hasattr(torch, 'jit') and self.device == "cuda":
            try:
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            except Exception as e:
                print(f"Warning: Could not enable CUDA optimizations: {e}")

        print("Model loaded successfully!")

    def _optimize_speaker_conditioning(self, speaker_wav_path: str):
        """Optimize speaker conditioning with proper memory management."""
        print("Computing speaker conditioning latents...")

        with torch.no_grad():
            self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
                audio_path=speaker_wav_path,
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len,
                sound_norm_refs=self.config.sound_norm_refs,
            )

            # Optimize memory usage based on device
            if self.device == "cuda":
                # Tensors are already on GPU, just ensure they're contiguous for efficiency
                self.gpt_cond_latent = self.gpt_cond_latent.contiguous()
                self.speaker_embedding = self.speaker_embedding.contiguous()
            else:
                # For CPU tensors, pin memory for faster GPU transfer if needed
                try:
                    self.gpt_cond_latent = self.gpt_cond_latent.pin_memory()
                    self.speaker_embedding = self.speaker_embedding.pin_memory()
                except RuntimeError:
                    # If pinning fails, just use tensors as-is
                    pass

        print("Speaker conditioning computed successfully!")

    @contextmanager
    def _timing_context(self, description: str):
        """Context manager for timing operations."""
        start_time = time.time()
        yield
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{description}: {elapsed:.2f} seconds")
        self.total_processing_time += elapsed

    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _group_by_language(self, df: pd.DataFrame) -> Dict[str, List[Tuple[int, str, str]]]:
        """Group sentences by language for efficient processing."""
        language_groups = defaultdict(list)

        for idx, row in df.iterrows():
            language_full = row["language"].strip()
            text = row["generated_sentence"].strip()
            sentence_id = str(row["sentence_id"]).strip()

            # Validate language
            if language_full not in self.language_mapping:
                print(f"Warning: Unknown language '{language_full}' for sentence {sentence_id}. Skipping.")
                continue

            # Skip empty text
            if not text:
                print(f"Warning: Empty text for sentence {sentence_id}. Skipping.")
                continue

            language_code = self.language_mapping[language_full]
            # language_code = language_full

            language_groups[language_code].append((idx, text, sentence_id))

        return language_groups

    def _single_inference(
        self, 
        text: str, 
        language: str, 
        **generation_params
    ) -> Optional[torch.Tensor]:
        """Perform single inference."""
        try:
            with torch.no_grad():
                wav_chunk = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    **generation_params
                )
                return torch.tensor(wav_chunk["wav"]).unsqueeze(0)
        except Exception as e:
            print(f"Error in single inference for text '{text[:50]}...': {str(e)}")
            return None

    def _save_audio_with_retry(self, audio_tensor: torch.Tensor, output_path: Path, max_retries: int = 3):
        """Save audio with retry logic."""
        for attempt in range(max_retries):
            try:
                torchaudio.save(
                    str(output_path),  # Convert Path to string for compatibility
                    audio_tensor,
                    sample_rate=self.config.audio.output_sample_rate,
                )
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to save audio after {max_retries} attempts: {str(e)}")
                    return False
                time.sleep(0.1)  # Brief pause before retry
        return False

    def process_language_group(
        self, 
        language_code: str, 
        items: List[Tuple[int, str, str]], 
        output_folder: Path,
        batch_size: int = 4,
        **generation_params
    ) -> Tuple[int, int]:
        """Process a group of items with the same language."""
        successful = 0
        failed = 0

        # Create language-specific subfolder
        language_output_folder = output_folder / language_code
        language_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created language folder: {language_output_folder}")

        with self._timing_context(f"Processing {len(items)} items in {language_code}"):
            # Process items individually for better error handling
            for idx, text, sentence_id in tqdm(items, desc=f"Processing {language_code}"):
                audio_tensor = self._single_inference(text, language_code, **generation_params)

                if audio_tensor is not None:
                    # Save in language-specific subfolder
                    output_path = language_output_folder / f"{sentence_id}.wav"
                    if self._save_audio_with_retry(audio_tensor, output_path):
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1

                # Periodic memory cleanup
                if (successful + failed) % 10 == 0:
                    self._clear_gpu_memory()

        return successful, failed

    def process_csv_batch(
        self,
        csv_path: str,
        speaker_wav_path: str,
        output_dir: str,
        batch_size: int = 4,
        max_workers: int = 2,
        use_async: bool = False,  # Disable async by default for stability
        temperature: float = 0.75,
        length_penalty: float = 1.0,
        repetition_penalty: float = 5.0,
        top_k: int = 50,
        top_p: float = 0.85
    ):
        """Process a CSV file for optimized batch inference."""

        # Read and validate CSV
        df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8')
        print(f"Processing {len(df)} samples from {csv_path}")

        # Validate CSV columns
        required_columns = ["language", "generated_sentence", "sentence_id"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Create output directory
        csv_name = Path(csv_path).stem
        output_folder = Path(output_dir) / csv_name
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {output_folder}")
        print("Language-specific subfolders will be created automatically.")

        # Optimize speaker conditioning
        self._optimize_speaker_conditioning(speaker_wav_path)

        # Clear GPU memory before processing
        self._clear_gpu_memory()

        # Group by language for efficient processing
        with self._timing_context("Grouping sentences by language"):
            language_groups = self._group_by_language(df)

        print(f"Found {len(language_groups)} languages: {list(language_groups.keys())}")
        for lang, items in language_groups.items():
            print(f"  {lang}: {len(items)} sentences")

        # Generation parameters
        generation_params = {
            "temperature": temperature,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "top_p": top_p
        }

        # Process batches
        start_time = time.time()

        # Sequential processing (more stable)
        print("Using sequential processing...")
        successful_count = 0
        failed_count = 0

        for language_code, items in language_groups.items():
            lang_successful, lang_failed = self.process_language_group(
                language_code, 
                items, 
                output_folder, 
                batch_size,
                **generation_params
            )
            successful_count += lang_successful
            failed_count += lang_failed

        # Final cleanup
        self._clear_gpu_memory()

        # Calculate and display performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        total_samples = successful_count + failed_count

        print(f"\n{'='*50}")
        print("BATCH PROCESSING COMPLETED!")
        print(f"{'='*50}")
        print(f"Total samples processed: {total_samples}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        if total_samples > 0:
            print(f"Success rate: {(successful_count/total_samples)*100:.1f}%")
        print(f"Total processing time: {total_time:.2f} seconds")

        if successful_count > 0:
            print(f"Average time per sample: {total_time/total_samples:.2f} seconds")
            print(f"Throughput: {successful_count/total_time:.2f} samples/second")

        print(f"\nOutput structure:")
        print(f"├── {output_folder}")
        for lang_code in language_groups.keys():
            lang_folder = output_folder / lang_code
            if lang_folder.exists():
                wav_files = list(lang_folder.glob("*.wav"))
                print(f"│   ├── {lang_code}/ ({len(wav_files)} files)")

        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized batch inference using tuned XTTS model with language-specific folders"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with columns: language, generated_sentence, sentence_id"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to model config.json file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to vocab.json file"
    )
    parser.add_argument(
        "--speaker_wav",
        type=str,
        required=True,
        help="Path to reference speaker audio file (.wav)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./batch_inference_output",
        help="Output directory for generated audio files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Length penalty for generation"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=10.0,
        help="Repetition penalty for generation"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.3,
        help="Top-p (nucleus) sampling parameter"
    )

    args = parser.parse_args()

    # Validate input files
    required_files = [
        (args.csv_path, "CSV file"),
        (args.config_path, "Config file"),
        (args.checkpoint_path, "Checkpoint file"),
        (args.vocab_path, "Vocab file"),
        (args.speaker_wav, "Speaker audio file")
    ]

    for file_path, file_type in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} not found: {file_path}")

    # Set device
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"Warning: Requested device '{args.device}' not available. Using '{device}' instead.")

    # Initialize inference system
    inference_system = OptimizedXTTSBatchInference(
        args.config_path,
        args.checkpoint_path,
        args.vocab_path,
        device
    )

    # Process batch
    inference_system.process_csv_batch(
        csv_path=args.csv_path,
        speaker_wav_path=args.speaker_wav,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        temperature=args.temperature,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p
    )


if __name__ == "__main__":
    main()


# Example usage:
# python -m scripts.inference.batch_infer_tuned \
#     --csv_path "benchmark/transliterated_mixed_code.csv" \
#     --config_path "checkpoints/GPT_XTTS_FT_MULTILINGUAL-June-27-2025_10+31AM-d078f73/config.json" \
#     --checkpoint_path "checkpoints/GPT_XTTS_FT_MULTILINGUAL-June-27-2025_10+31AM-d078f73/best_model.pth" \
#     --vocab_path "checkpoints/XTTS_v2.0_original_model_files/vocab.json" \
#     --speaker_wav "palki-ref.wav" \
#     --output_dir "./batch_inference_output" \
#     --batch_size 4 \