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

print("Dataset Tamil loaded successfully.")

# load malayalam dataset
ds = load_dataset(
    "ai4bharat/IndicVoices",
    "malayalam",
    token=HF_TOKEN,
    cache_dir=download_dir
)

print("Dataset Malayalam loaded successfully.")
