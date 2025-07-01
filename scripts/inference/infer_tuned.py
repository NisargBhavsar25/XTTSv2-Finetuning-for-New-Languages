import torch
import torchaudio
from tqdm import tqdm
# from underthesea import sent_tokenize  # For Vietnamese text processing
# add root path to sys.path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
timestamp = "May-23-2025_01+20PM-8e59ec3"
# Model paths
xtts_checkpoint = f"/home/ubuntu/Dikshit/XTTSv2-Fine-Tuning-Multilingual-combined-tokens/checkpoints/GPT_XTTS_FT_MULTILINGUAL-June-30-2025_08+48AM-267c788/best_model_41824.pth"
xtts_config = f"/home/ubuntu/Dikshit/XTTSv2-Fine-Tuning-Multilingual-combined-tokens/checkpoints/GPT_XTTS_FT_MULTILINGUAL-June-30-2025_08+48AM-267c788/config.json"
xtts_vocab = "/home/ubuntu/Dikshit/XTTSv2-Fine-Tuning-Multilingual-combined-tokens/checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab)
model.to(device)

print("Model loaded successfully!")

# Get voice conditioning from reference audio
speaker_audio_file = "/home/ubuntu/Dikshit/XTTSv2-Fine-Tuning-Multilingual-combined-tokens/Testing.wav"
language = "bn"  # Gujarati language code

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
    "NoBroker-এ বাড়ি খোঁজা অবিশ্বাস্যরকম সহজ এবং সুবিধাজনক কারণ আপনাকে কোনও ব্রোকারেজ ফি দিতে হবে না এবং আপনি সরাসরি সম্পত্তির মালিকদের সাথে যোগাযোগ করতে পারবেন, যা আপনার সময় এবং অর্থ উভয়ই সাশ্রয় করে এবং স্বচ্ছ এবং ঝামেলামুক্ত বাড়ি খোঁজার অভিজ্ঞতা নিশ্চিত করে।" #bengali
    # "NoBroker પર ઘર શોધવું અતિ સરળ અને અનુકૂળ છે કારણ કે તમારે કોઈ બ્રોકરેજ ફી ચૂકવવાની જરૂર નથી, અને તમે સીધા જ મિલકત માલિકો સાથે જોડાઈ શકો છો, જે પારદર્શક અને મુશ્કેલી-મુક્ત ઘર શોધ અનુભવ સુનિશ્ચિત કરતી વખતે તમારો સમય અને પૈસા બંને બચાવે છે." # Gujarati
    # "Finding a house on NoBroker is incredibly easy and convenient because you don't have to pay any brokerage fees, and you can directly connect with property owners, which saves you both time and money while ensuring a transparent and hassle-free house hunting experience."
    # "நோப்ரோக்கர்-ல் வீடு தேடுவது மிகவும் எளிது மற்றும் வசதியானது, ஏனென்றால் இங்கே எந்த புரோக்கரேஜ் கட்டணமும் கொடுக்க வேண்டியதில்லை மற்றும் நீங்கள் நேரடியாக வீட்டின் உரிமையாளருடன் தொடர்பு கொள்ளலாம், இதனால் உங்கள் நேரமும் பணமும் மிச்சமாகிறது." # tamil
    # "NoBroker से house ढूंढना इतना आसान और easy है, because यहाँ आपको कोई ब्रोकरेज फीस नहीं देनी पड़ती और आप सीधे मकान मालिक से बात कर सकते हैं, जिससे आपका समय और पैसा दोनों बचता है।"
    # "നോബ്രോക്കർ-ൽ വീട് കണ്ടെത്തുന്നത് വളരെ എളുപ്പവും സൗകര്യപ്രദവുമാണ്, കാരണം ഇവിടെ ബ്രോക്കറേജ് ഫീസ് ഒന്നും നൽകേണ്ടതില്ല, കൂടാതെ നിങ്ങൾക്ക് നേരിട്ട് വീട്ടുടമയുമായി ബന്ധപ്പെടാൻ കഴിയും, ഇത് നിങ്ങളുടെ സമയവും പണവും ലാഭിക്കാൻ സഹായിക്കുന്നു." # Malayalam
    # "तू मला मदत करशील का?"  # Marathi
    # "ನೋಬ್ರೋಕರ್ ನಲ್ಲಿ ಮನೆ ಹುಡುಕುವುದು ತುಂಬಾ ಸುಲಭ ಮತ್ತು ಅನುಕೂಲಕರವಾಗಿದೆ, ಏಕೆಂದರೆ ಇಲ್ಲಿ ಯಾವುದೇ ಬ್ರೋಕರೇಜ್ ಫೀಸು ನೀಡಬೇಕಾಗಿಲ್ಲ ಮತ್ತು ನೀವು ನೇರವಾಗಿ ಮನೆ ಮಾಲೀಕರೊಂದಿಗೆ ಸಂಪರ್ಕ ಸಾಧಿಸಬಹುದು, ಇದರಿಂದ ನಿಮ್ಮ ಸಮಯ ಮತ್ತು ಹಣ ಎರಡೂ ಉಳಿತಾಯವಾಗುತ್ತದೆ." # Kannada
    # "నోబ్రోకర్ లో ఇల్లు వెతకడం చాలా సులభం మరియు సౌకర్యవంతమైనది, ఎందుకంటే ఇక్కడ ఎలాంటి బ్రోకరేజ్ ఫీసు చెల్లించాల్సిన అవసరం లేదు మరియు మీరు నేరుగా ఇంటి యజమానితో సంప్రదించవచ్చు, దీని వలన మీ సమయం మరియు డబ్బు రెండూ ఆదా అవుతాయి." # Telugu
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
torchaudio.save("output_bengali_sep_token.wav", out_wav, 24000)

# # For Jupyter Notebook, play the audio
# from IPython.display import Audio
# Audio(out_wav, rate=24000)