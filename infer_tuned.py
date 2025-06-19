import torch
import torchaudio
from tqdm import tqdm
# from underthesea import sent_tokenize  # For Vietnamese text processing

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# timestamp = "May-23-2025_01+20PM-8e59ec3"
# Model paths
xtts_checkpoint = f"checkpoints/GPT_XTTS_FT-May-29-2025_12+48PM-2410589/checkpoint_143000.pth"
xtts_config = f"checkpoints/GPT_XTTS_FT-May-29-2025_12+48PM-2410589/config.json"
xtts_vocab = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab)
model.to(device)

print("Model loaded successfully!")

# Get voice conditioning from reference audio
speaker_audio_file = "reference_as_2.wav"
language = "as"  # Gujarati language code

gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=30,
    max_ref_length=10,
    sound_norm_refs=False
)

# Text to synthesize
# tts_text = "પોતાનાં બાળકોને કયા પ્રકારનું શિક્ષણ આપવું તે પસંદ કરવાનો પ્રથમ અધિકાર માબાપોને રહેશે."  # "Parents will have the first right to choose what kind of education to give their children."

# For longer texts, split into sentences
# tts_texts = sent_tokenize(tts_text)
tts_texts = [
    "শাওন-ভাদ মাহ দুটা বর্ষা ঋতু বা বাৰিষা কাল। কিন্তু অসমত বহাগ মাহৰ পৰাই বৰষুণ আৰম্ভ হৈ একেবাৰে আহিন. মাহলৈকে থাকে ৷ ইয়াৰ ভিতৰত জেঠ, আহাৰ, শাওন আৰু ভাদ এই চাৰি মাহত বাৰিষাৰ প্ৰকোপ আটাইতকৈ চৰে"
]

# Process each sentence
wav_chunks = []
for text in tqdm(tts_texts):
    output = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(output["wav"]))

# Combine the outputs
out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Save the audio
torchaudio.save("output_as_2.wav", out_wav, 24000)

# # For Jupyter Notebook, play the audio
# from IPython.display import Audio
# Audio(out_wav, rate=24000)