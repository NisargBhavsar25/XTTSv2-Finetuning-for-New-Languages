import glob
import os
import random
from multiprocessing import Manager

import numpy as np
import torch
from torch.utils.data import Dataset


class GPTGANDataset(Dataset):
    """
    GAN Dataset searchs for all the wav files under root path
    and converts them to acoustic features on the fly and returns
    random segments of (audio, feature) couples.
    """

    def __init__(
        self,
        ap,
        items,
        seq_len,
        hop_len,
        pad_short,
        conv_pad=2,
        ar_mel_length_compression = 1024,
        output_hop_length = 256,
        output_sample_rate = 24000,
        input_sample_rate = 22050,
        return_pairs=False,
        is_training=True,
        return_segments=True,
        use_noise_augment=False,
        use_cache=False,
        verbose=False,
        train_spk_encoder=False
    ):
        super().__init__()
        self.ap = ap
        self.item_list = items
        self.compute_feat = not isinstance(items[0], (tuple, list))
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.pad_short = pad_short
        self.conv_pad = conv_pad
        self.ar_mel_length_compression = ar_mel_length_compression
        self.output_hop_length = output_hop_length
        self.output_sample_rate = output_sample_rate
        self.input_sample_rate = input_sample_rate
        self.return_pairs = return_pairs
        self.is_training = is_training
        self.return_segments = return_segments
        self.use_cache = use_cache
        self.use_noise_augment = use_noise_augment
        self.verbose = verbose
        self.train_spk_encoder = train_spk_encoder

        assert seq_len % hop_len == 0, " [!] seq_len has to be a multiple of hop_len."
        self.feat_frame_len = seq_len // hop_len + (2 * conv_pad)

        # map G and D instances
        self.G_to_D_mappings = list(range(len(self.item_list)))
        self.shuffle_mapping()

        # cache acoustic features
        if use_cache:
            self.create_feature_cache()

    def create_feature_cache(self):
        self.manager = Manager()
        self.cache = self.manager.list()
        self.cache += [None for _ in range(len(self.item_list))]

    @staticmethod
    def find_wav_files(path):
        return glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        """Return different items for Generator and Discriminator and
        cache acoustic features"""

        # set the seed differently for each worker
        if torch.utils.data.get_worker_info():
            random.seed(torch.utils.data.get_worker_info().seed)

        if self.return_segments:
            item1 = self.load_item(idx)
            if self.return_pairs:
                idx2 = self.G_to_D_mappings[idx]
                item2 = self.load_item(idx2)
                return item1, item2
            return item1
        item1 = self.load_item(idx)
        return item1

    def _pad_short_samples(self, audio, mel=None):
        """Pad samples shorter than the output sequence length"""
        if len(audio) < self.seq_len:
            audio = np.pad(audio, (0, self.seq_len - len(audio)), mode="constant", constant_values=0.0)
            
        if mel is not None and mel.shape[1] < self.feat_frame_len:
            pad_value = self.ap.melspectrogram(np.zeros([self.ap.win_length]))[:, 0]
            mel = np.pad(
                mel,
                ([0, 0], [0, self.feat_frame_len - mel.shape[1]]),
                mode="constant",
                constant_values=pad_value.mean(),
            )
        return audio, mel

    def shuffle_mapping(self):
        random.shuffle(self.G_to_D_mappings)
    
    def interpolate(self, latents):
        latents = torch.from_numpy(latents).unsqueeze(0)
        z = torch.nn.functional.interpolate(
            latents,
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length],
            mode="linear",
        ).squeeze(1)
        # upsample to the right sr
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[self.output_sample_rate / self.input_sample_rate],
                mode="linear",
            ).squeeze(0)
        
        return z.cpu().numpy()

    def load_item(self, idx):
        """load (audio, feat) couple"""
        if self.compute_feat:
            # compute features from wav
            wavpath = self.item_list[idx]
            # print(wavpath)

            if self.use_cache and self.cache[idx] is not None:
                audio, mel = self.cache[idx]
            else:
                audio = self.ap.load_wav(wavpath)
                mel = self.ap.melspectrogram(audio)
                audio, mel = self._pad_short_samples(audio, mel)
        else:
            # load precomputed features
            wavpath, feat_path, spk_path = self.item_list[idx]

            if self.use_cache and self.cache[idx] is not None:
                audio, mel = self.cache[idx]
            else:
                raw_audio = self.ap.load_wav(wavpath)
                mel = np.load(feat_path)
                spk = np.load(spk_path)
                mel = self.interpolate(mel)
                audio, mel = self._pad_short_samples(raw_audio, mel)

        # correct the audio length wrt padding applied in stft
        if len(audio) > mel.shape[-1] * self.hop_len:
            audio = np.pad(audio, (0, self.hop_len), mode="edge")
        else:
            audio = np.pad(audio, (0, mel.shape[-1] * self.hop_len - len(audio)), mode="edge")
        audio = audio[: mel.shape[-1] * self.hop_len]

        assert (
            mel.shape[-1] * self.hop_len == audio.shape[-1]
        ), f" [!] {mel.shape[-1] * self.hop_len} vs {audio.shape[-1]}"

        audio = torch.from_numpy(audio).float().unsqueeze(0)
        mel = torch.from_numpy(mel).float().squeeze(0)

        if self.return_segments:
            max_mel_start = mel.shape[1] - self.feat_frame_len
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.feat_frame_len
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hop_len
            audio = audio[:, audio_start : audio_start + self.seq_len]

        if self.use_noise_augment and self.is_training and self.return_segments:
            audio = audio + (1 / 32768) * torch.randn_like(audio)

        if self.train_spk_encoder:
            return {"mel": mel, "audio": audio, "raw_audio": raw_audio}
        else:
            return (mel, audio, spk)
    
    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        # stack for features that already have the same shape
        batch["mel"] = torch.stack(batch["mel"])
        batch["audio"] = torch.stack(batch["audio"])

        wav_len = [wav.shape[0] for wav in batch["raw_audio"]]
        max_wav_len = max(wav_len)

        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        wav_padded = wav_padded.zero_()
        for i in range(B):
            wav = batch["raw_audio"][i]
            wav_padded[i, :, : batch["raw_audio"][i].shape[0]] = torch.FloatTensor(wav)
        batch["wav"] = wav_padded

        return batch