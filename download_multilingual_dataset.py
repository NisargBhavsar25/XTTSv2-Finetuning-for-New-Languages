from datasets import load_dataset
import dotenv
import os
import soundfile as sf
from tqdm.auto import tqdm
import time

# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Verify token exists
if not HF_TOKEN:
    print("❌ Error: HUGGINGFACE_TOKEN not found in environment variables")
    print("Please create a .env file with your Hugging Face token:")
    print("HUGGINGFACE_TOKEN=your_token_here")
    exit(1)

print(f"✅ Token loaded: {HF_TOKEN[:10]}...")

# Set download directory
download_dir = "./dataset_cache"
os.makedirs(download_dir, exist_ok=True)

# Define the languages we want to process
languages = ["tamil", "malayalam"]
target_hours_per_language = 200
target_seconds_per_language = target_hours_per_language * 3600

# Define the base directory for saving datasets
base_dir = "./datasets-multilingual"
wavs_dir = os.path.join(base_dir, "wavs")

# Create directories with absolute path verification
def create_directories():
    try:
        abs_base_dir = os.path.abspath(base_dir)
        abs_wavs_dir = os.path.abspath(wavs_dir)
        
        os.makedirs(abs_base_dir, exist_ok=True)
        os.makedirs(abs_wavs_dir, exist_ok=True)
        
        print(f"✅ Created directories:")
        print(f"   Base: {abs_base_dir}")
        print(f"   Wavs: {abs_wavs_dir}")
        
        # Test write permissions
        test_file = os.path.join(abs_base_dir, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Write permissions confirmed")
        
        return abs_base_dir, abs_wavs_dir
    except Exception as e:
        print(f"❌ Error creating directories: {str(e)}")
        return None, None

def test_dataset_access():
    """Test if we can access the IndicVoices dataset"""
    try:
        print("🔍 Testing dataset access...")
        test_ds = load_dataset(
            "ai4bharat/IndicVoices",
            "tamil",
            split="train[:1]",  # Load just 1 sample for testing
            token=HF_TOKEN,
            cache_dir=download_dir
        )
        print(f"✅ Dataset access successful! Sample keys: {list(test_ds[0].keys())}")
        return True
    except Exception as e:
        print(f"❌ Dataset access failed: {str(e)}")
        return False

def save_multilingual_dataset(languages, abs_base_dir, abs_wavs_dir, target_seconds_per_language):
    # Test dataset access first
    if not test_dataset_access():
        print("❌ Cannot access dataset. Please check your token and internet connection.")
        return
    
    # Prepare metadata lists
    train_metadata = []
    eval_metadata = []
    
    # Stats tracking
    total_processed = 0
    total_saved_files = 0
    start_time = time.time()
    
    # Process each language
    for lang_idx, lang in enumerate(languages):
        print(f"\n{'='*60}")
        print(f"Processing language {lang_idx+1}/{len(languages)}: {lang.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load the dataset for this language
            print(f"📥 Loading dataset for {lang}...")
            ds = load_dataset(
                "ai4bharat/IndicVoices",
                lang,
                token=HF_TOKEN,
                cache_dir=download_dir
            )
            print(f"✅ Dataset loaded for {lang}. Available splits: {list(ds.keys())}")
            
            lang_processed = 0
            lang_duration = 0
            lang_saved_files = 0
            
            # Process train split first (usually largest)
            splits_to_process = ["train"] + [s for s in ds.keys() if s != "train"]
            
            for split in splits_to_process:
                if lang_duration >= target_seconds_per_language:
                    print(f"🎯 Target duration reached for {lang}")
                    break
                
                if split not in ds:
                    continue
                    
                split_data = ds[split]
                print(f"\n📊 Processing {lang}-{split}: {len(split_data)} samples")
                
                split_saved = 0
                split_duration = 0
                
                # Process samples with progress bar
                with tqdm(total=len(split_data), desc=f"{lang}-{split}", unit="samples") as pbar:
                    
                    for idx in range(len(split_data)):
                        if lang_duration >= target_seconds_per_language:
                            break
                            
                        try:
                            example = split_data[idx]
                            
                            # Extract audio data
                            audio_array = None
                            sampling_rate = None
                            
                            # Try different possible audio field names
                            audio_fields = ['audio', 'audio_filepath', 'sound', 'wav']
                            for field in audio_fields:
                                if field in example:
                                    audio_data = example[field]
                                    if isinstance(audio_data, dict) and 'array' in audio_data:
                                        audio_array = audio_data['array']
                                        sampling_rate = audio_data['sampling_rate']
                                        break
                                    else:
                                        try:
                                            audio_array, sampling_rate = sf.read(audio_data)
                                            break
                                        except:
                                            continue
                            
                            if audio_array is None:
                                pbar.update(1)
                                continue
                            
                            # Calculate duration
                            audio_duration = len(audio_array) / sampling_rate
                            
                            # Skip very short audio
                            if audio_duration < 0.5:
                                pbar.update(1)
                                continue
                            
                            # Check if we would exceed target
                            if lang_duration + audio_duration > target_seconds_per_language:
                                remaining = target_seconds_per_language - lang_duration
                                if remaining < 10:  # Less than 10 seconds remaining
                                    break
                            
                            # Save audio file
                            file_name = f"{lang}_{split}_{idx:06d}.wav"
                            file_path = os.path.join(abs_wavs_dir, file_name)
                            
                            try:
                                sf.write(file_path, audio_array, sampling_rate)
                                
                                # Verify file was saved
                                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                                    # Create metadata entry
                                    relative_path = f"wavs/{file_name}"
                                    text = example.get("text", "").strip()
                                    speaker = example.get("speaker_id", "unknown")
                                    entry = f"{relative_path}|{text}|{speaker}|{lang}"
                                    
                                    if split == "train":
                                        train_metadata.append(entry)
                                    else:
                                        eval_metadata.append(entry)
                                    
                                    # Update counters
                                    lang_processed += 1
                                    lang_duration += audio_duration
                                    lang_saved_files += 1
                                    split_saved += 1
                                    split_duration += audio_duration
                                    
                                    # Update progress bar with current stats
                                    pbar.set_postfix(
                                        saved=split_saved,
                                        duration=f"{split_duration/3600:.2f}h",
                                        total_duration=f"{lang_duration/3600:.2f}h"
                                    )
                                
                            except Exception as save_e:
                                print(f"❌ Save error: {save_e}")
                        
                        except Exception as process_e:
                            print(f"❌ Process error: {process_e}")
                        
                        pbar.update(1)
                
                print(f"✅ {lang}-{split}: {split_saved} files saved, {split_duration/3600:.2f}h")
            
            total_processed += lang_processed
            total_saved_files += lang_saved_files
            
            print(f"🎉 {lang} completed: {lang_saved_files} files, {lang_duration/3600:.2f}h")
            
            # Verify files in directory
            actual_files = len([f for f in os.listdir(abs_wavs_dir) if f.startswith(lang)])
            print(f"📁 Verified {actual_files} files in directory for {lang}")
            
        except Exception as e:
            print(f"❌ Failed to process {lang}: {str(e)}")
            continue
    
    # Save metadata files
    print(f"\n{'='*40}")
    print("💾 Saving metadata files...")
    
    try:
        train_file = os.path.join(abs_base_dir, "metadata_train.csv")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("audio_file|text|speaker_name|language\n")
            f.write("\n".join(train_metadata))
        print(f"✅ Train metadata: {len(train_metadata)} entries")
        
        eval_file = os.path.join(abs_base_dir, "metadata_eval.csv")
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write("audio_file|text|speaker_name|language\n")
            f.write("\n".join(eval_metadata))
        print(f"✅ Eval metadata: {len(eval_metadata)} entries")
        
    except Exception as e:
        print(f"❌ Metadata save error: {e}")
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n🏁 FINAL SUMMARY:")
    print(f"⏱️  Processing time: {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"📊 Total files saved: {total_saved_files}")
    print(f"📁 Files per language:")
    
    if os.path.exists(abs_wavs_dir):
        for lang in languages:
            count = len([f for f in os.listdir(abs_wavs_dir) if f.startswith(lang)])
            print(f"   {lang}: {count} files")

# Run the process
print("🚀 Starting multilingual dataset creation...")
abs_base_dir, abs_wavs_dir = create_directories()

if abs_base_dir and abs_wavs_dir:
    save_multilingual_dataset(languages, abs_base_dir, abs_wavs_dir, target_seconds_per_language)
else:
    print("❌ Cannot create directories. Process aborted.")