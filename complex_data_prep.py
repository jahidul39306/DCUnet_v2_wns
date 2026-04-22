import os
import torch
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf


class ComplexSpeechDataset(Dataset):
    def __init__(
        self,
        clean_dir,
        noisy_dir,
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        max_duration_sec=4.0,
    ):
        super().__init__()
        self.clean_files = sorted(Path(clean_dir).rglob("*.wav"))
        self.noisy_files = sorted(Path(noisy_dir).rglob("*.wav"))

        # They must be the same length and matched by sorted order (or filename)
        assert len(self.clean_files) == len(
            self.noisy_files
        ), "Clean and noisy dirs must have the same number of files!"

        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_sec)
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
        )

    def __len__(self):
        return len(self.clean_files)

    def _pad_or_truncate(self, waveform):
        if waveform.shape[1] > self.max_length:
            start = random.randint(0, waveform.shape[1] - self.max_length)
            waveform = waveform[:, start : start + self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def _load_audio(self, path):
        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
             waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(
                waveform
            )
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

            max_amp = torch.max(torch.abs(waveform))

            waveform = waveform / (max_amp + 1e-8)
        return self._pad_or_truncate(waveform)

    def __getitem__(self, idx):
        clean_waveform = self._load_audio(self.clean_files[idx])
        noisy_waveform = self._load_audio(self.noisy_files[idx])  # paired file

        eps = 1e-8
        # Normalize by the noisy mixture (the input), same as before
        max_val = torch.max(torch.abs(noisy_waveform)) + eps
        noisy_waveform = noisy_waveform / max_val
        clean_waveform = clean_waveform / max_val  # same scale as noisy

        mix_stft = self.stft(noisy_waveform)
        clean_stft = self.stft(clean_waveform)

        normalize_factor = torch.max(torch.abs(mix_stft)) + eps
        mix_real = mix_stft.real / normalize_factor
        mix_imag = mix_stft.imag / normalize_factor
        clean_real = clean_stft.real / normalize_factor
        clean_imag = clean_stft.imag / normalize_factor

        return mix_real, mix_imag, clean_real, clean_imag
    
def get_dataloaders(clean_dir, noisy_dir, batch_size=16):
    dataset = ComplexSpeechDataset(clean_dir, noisy_dir)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader
