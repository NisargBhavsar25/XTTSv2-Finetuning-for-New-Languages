import os
import wave
import numpy as np

def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

# Folder path - update this to your wav folder location if needed
folder_path = 'datasets-gj/wavs'

# Lists to hold durations
train_durations = []
valid_durations = []
short_durations = []

# Threshold for short samples in seconds (based on previous recommendations)
short_threshold = 3.0

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(folder_path, filename)
        duration = get_wav_duration(file_path)
        
        if filename.startswith('train_'):
            train_durations.append(duration)
            if duration <= short_threshold:
                short_durations.append(duration)
        elif filename.startswith('valid_'):
            valid_durations.append(duration)
            if duration <= short_threshold:
                short_durations.append(duration)

# Calculate total durations in hours
train_hours = sum(train_durations) / 3600
valid_hours = sum(valid_durations) / 3600
short_hours = sum(short_durations) / 3600

# Calculate statistics on sample counts
train_count = len(train_durations)
valid_count = len(valid_durations)
short_count = len(short_durations)
total_count = train_count + valid_count

print(f"Training dataset: {train_hours:.2f} hours ({train_count} files)")
print(f"Validation dataset: {valid_hours:.2f} hours ({valid_count} files)")
print(f"Short sample dataset (≤{short_threshold}s): {short_hours:.2f} hours ({short_count} files)")
print(f"Short samples make up {short_hours/(train_hours+valid_hours)*100:.1f}% of total dataset duration")

if short_hours >= 20:
    print("\nYou have sufficient short sample data (≥20 hours) for XTTSv2 fine-tuning!")
    print("According to the repository guidance, you do not need to fine-tune the DVAE.")
else:
    print(f"\nYou have {short_hours:.2f} hours of short samples, which is less than the recommended 20 hours.")
    print("You might need to fine-tune the DVAE or collect more short samples.")
