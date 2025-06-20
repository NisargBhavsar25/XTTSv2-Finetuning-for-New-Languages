from datasets import load_dataset
import dotenv
import os
import soundfile as sf
import numpy as np
import io
from tqdm.auto import tqdm
import time

# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Set download directory
download_dir = "dataset_cache/"

# Load the dataset
ds = load_dataset(
    "ai4bharat/IndicVoices",
    "tamil",
    token=HF_TOKEN,
    cache_dir=download_dir
)

# Define the base directory for saving datasets
base_dir = "datasets-ta"
wavs_dir = os.path.join(base_dir, "wavs")

# Create directories if they don't exist
os.makedirs(wavs_dir, exist_ok=True)

# Function to save audio files and create metadata
def save_dataset(ds, base_dir, wavs_dir):
    # Prepare metadata lists
    train_metadata = []
    eval_metadata = []
    
    # Stats tracking
    total_processed = 0
    total_errors = 0
    start_time = time.time()
    
    # Process each split
    for split in ds.keys():
        print(f"\n[{time.strftime('%H:%M:%S')}] Processing split '{split}' with {len(ds[split])} samples...")
        split_errors = 0
        
        # Process in batches to avoid memory issues
        batch_size = 100
        total_samples = len(ds[split])
        
        # Create progress bar for the entire split
        with tqdm(total=total_samples, desc=f"Processing {split}", unit="samples") as pbar:
            batch_index = 0
            while batch_index < total_samples:
                batch_start = batch_index
                batch_end = min(batch_start + batch_size, total_samples)
                
                # Try to get a batch, skip it entirely if it fails
                try:
                    batch = ds[split].select(range(batch_start, batch_end))
                    
                    # Process each sample in the batch, with careful error handling
                    batch_success = 0
                    batch_items = []
                    
                    # Safely get all examples from batch
                    try:
                        batch_items = list(enumerate(batch))
                    except Exception as e:
                        tqdm.write(f"❌ Error iterating batch {batch_start}-{batch_end}: {str(e)}")
                        split_errors += (batch_end - batch_start)
                        total_errors += (batch_end - batch_start)
                        pbar.update(batch_end - batch_start)
                        batch_index = batch_end
                        continue
                    
                    # Process each item individually
                    for i, example in batch_items:
                        global_idx = batch_start + i
                        
                        try:
                            # Handle the audio data safely
                            if 'audio_filepath' in example:
                                # Access the nested audio data structure
                                if isinstance(example['audio_filepath'], dict) and 'array' in example['audio_filepath']:
                                    audio_array = example['audio_filepath']['array']
                                    sampling_rate = example['audio_filepath']['sampling_rate']
                                else:
                                    # If it's a path or BytesIO object, try to read it directly
                                    file_obj = example['audio_filepath']
                                    audio_array, sampling_rate = sf.read(file_obj)
                            elif 'audio' in example:
                                # Alternative structure
                                if isinstance(example['audio'], dict) and 'array' in example['audio']:
                                    audio_array = example['audio']['array']
                                    sampling_rate = example['audio']['sampling_rate']
                                else:
                                    file_obj = example['audio']
                                    audio_array, sampling_rate = sf.read(file_obj)
                            
                            # Define file name with split and index
                            file_name = f"{split}_{global_idx}.wav"
                            file_path = os.path.join(wavs_dir, file_name)
                            
                            # Save audio file in wav format
                            sf.write(file_path, audio_array, sampling_rate)
                            
                            # Prepare metadata entry
                            relative_path = f"wavs/{file_name}"
                            text = example["text"]
                            speaker = example["speaker_id"]
                            entry = f"{relative_path}|{text}|{speaker}"
                            
                            # Append to appropriate metadata list
                            if split == "train":
                                train_metadata.append(entry)
                            elif split in ["valid", "eval", "test"]:
                                eval_metadata.append(entry)
                            
                            batch_success += 1
                            
                        except Exception as e:
                            split_errors += 1
                            total_errors += 1
                            if total_errors % 100 == 0:  # Limit error messages to avoid flooding terminal
                                tqdm.write(f"Error processing sample {global_idx} in {split} split: {str(e)}")
                            continue
                    
                    # Display batch completion message outside of progress bar
                    batch_size_actual = len(batch_items)
                    if batch_size_actual > 0:
                        tqdm.write(f"  Batch {batch_start//batch_size + 1}: {batch_success}/{batch_size_actual} samples processed successfully")
                
                except Exception as e:
                    tqdm.write(f"❌ Failed to process batch {batch_start}-{batch_end}: {str(e)}")
                    split_errors += (batch_end - batch_start)
                    total_errors += (batch_end - batch_start)
                
                # Update progress bar
                pbar.update(batch_end - batch_start)
                elapsed = time.time() - start_time
                success_rate = batch_success / batch_size if batch_size > 0 else 0
                pbar.set_postfix(
                    success_rate=f"{success_rate*100:.1f}%", 
                    elapsed=f"{int(elapsed//60)}m {int(elapsed%60)}s",
                    errors=total_errors
                )
                
                # Move to next batch
                batch_index = batch_end
                
                # Save progress periodically (every 1000 samples)
                if batch_index % 1000 == 0:
                    tqdm.write(f"💾 Saving intermediate metadata at sample {batch_index}...")
                    with open(os.path.join(base_dir, "metadata_train_partial.csv"), "w", encoding="utf-8") as f:
                        f.write("audio_file|text|speaker_name\n")
                        f.write("\n".join(train_metadata))
                    with open(os.path.join(base_dir, "metadata_eval_partial.csv"), "w", encoding="utf-8") as f:
                        f.write("audio_file|text|speaker_name\n")
                        f.write("\n".join(eval_metadata))
        
        # Report split completion
        processed_count = total_samples - split_errors
        total_processed += processed_count
        print(f"[{time.strftime('%H:%M:%S')}] Completed {split} split: {processed_count}/{total_samples} samples processed successfully")

    # Save metadata files
    print("\nSaving metadata files...")
    with open(os.path.join(base_dir, "metadata_train.csv"), "w", encoding="utf-8") as f:
        f.write("audio_file|text|speaker_name\n")
        f.write("\n".join(train_metadata))
    print(f"- Saved metadata_train.csv with {len(train_metadata)} entries")

    with open(os.path.join(base_dir, "metadata_eval.csv"), "w", encoding="utf-8") as f:
        f.write("audio_file|text|speaker_name\n")
        f.write("\n".join(eval_metadata))
    print(f"- Saved metadata_eval.csv with {len(eval_metadata)} entries")

    # Print overall statistics
    elapsed = time.time() - start_time
    print(f"\n[SUMMARY]")
    print(f"Total processing time: {int(elapsed//60)} minutes {int(elapsed%60)} seconds")
    print(f"Files successfully processed: {total_processed}")
    print(f"Errors encountered: {total_errors}")
    success_rate = total_processed / (total_processed + total_errors) * 100 if (total_processed + total_errors) > 0 else 0
    print(f"Success rate: {success_rate:.2f}%")
    print(f"All data saved to {base_dir} directory")

# Call the function
save_dataset(ds, base_dir, wavs_dir)
