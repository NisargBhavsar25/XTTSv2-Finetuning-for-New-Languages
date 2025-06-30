from datasets import load_dataset
from datasets import load_from_disk
import os
import soundfile as sf
import numpy as np

languages = ['Hindi', 'Kannada', 'Malayalam', 'Tamil', 'Telugu', 'Bengali', 'Gujarati', 'Marathi']
for lang in languages:
    dataset = load_dataset(f'SPRINGLab/IndicTTS_{lang}')
    dataset = dataset['train']

    # create a directory for all datasets
    if not os.path.exists('datasets-diff-wavs'):
        os.makedirs('datasets-diff-wavs')

    dataset.save_to_disk(f'datasets-diff-wavs/IndicTTS_{lang}')
    print(f'{lang} complete dataset downloaded')



for language in languages:
    print(f"Processing dataset for {language}...")
    # Create directories for saving wav files and metadata
    
    base_dir = 'datasets-diff-wavs'
    wavs_dir = os.path.join(base_dir, 'wavs-hindi')
    os.makedirs(wavs_dir, exist_ok=True)

    metadata = []

    # Load the dataset
    dataset_path = os.path.join(base_dir, f'IndicTTS_{language}')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset for {language} not found at {dataset_path}")
    try:
        data = load_from_disk(dataset_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset for {language}: {e}")

    # Process each item in the dataset
    for i, item in enumerate(data):
        try:
            # Extract audio data properly
            audio_array = np.array(item['audio']['array'])
            sampling_rate = item['audio']['sampling_rate']
            text = item['text']
            
            # Ensure audio array is in correct format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Create wav filename with language prefix
            wav_filename = f'{language}_wav_{i:05d}.wav'
            wav_path = os.path.join(wavs_dir, wav_filename)
            
            # Save audio as wav file
            sf.write(wav_path, audio_array, sampling_rate)
            
            # Add entry to metadata with language column
            metadata.append(f'wavs/{wav_filename}|{text}|{language}')
            
            # Print progress every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} files...")
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    # Write metadata CSV file with language-specific name
    metadata_path = os.path.join(base_dir, f'metadata_train_{language}.csv')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('audio_file|text|language\n')
        for line in metadata:
            f.write(line + '\n')

    print(f"Dataset conversion complete!")
    print(f"Created {len(metadata)} audio files in {wavs_dir}")
    print(f"Metadata saved to {metadata_path}")

    # Verify first file
    wav_files = [f for f in os.listdir(wavs_dir) if f.startswith(language)]
    if wav_files:
        first_file = os.path.join(wavs_dir, sorted(wav_files)[0])
        info = sf.info(first_file)
        print(f"First audio file info: {info}")

# combine all metadata files into one
combined_metadata_path = os.path.join(base_dir, 'metadata.csv')
with open(combined_metadata_path, 'w', encoding='utf-8') as outfile:
    outfile.write('audio_file|text|language\n')
    for lang in languages:
        metadata_path = os.path.join(base_dir, f'metadata_train_{lang}.csv')
        with open(metadata_path, 'r', encoding='utf-8') as infile:
            next(infile)  # skip header
            for line in infile:
                outfile.write(line)

print(f"Combined metadata saved to {combined_metadata_path}")

# copy all wav files to a single directory
combined_wavs_dir = os.path.join(base_dir, 'wavs')
os.makedirs(combined_wavs_dir, exist_ok=True)
for lang in languages:
    lang_wavs_dir = os.path.join(base_dir, f'wavs-{lang.lower()}')
    for wav_file in os.listdir(lang_wavs_dir):
        if wav_file.startswith(lang):
            src_path = os.path.join(lang_wavs_dir, wav_file)
            dst_path = os.path.join(combined_wavs_dir, wav_file)
            sf.write(dst_path, sf.read(src_path)[0], sf.info(src_path).samplerate)

print(f"All audio files copied to {combined_wavs_dir}")