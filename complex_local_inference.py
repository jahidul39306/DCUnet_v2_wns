import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchaudio
import soundfile as sf
import numpy as np
from complex_model import DeepComplexUNet

def infer_local_audio(noisy_wav_path, model_path, output_path="cleaned_result_local.wav"):
    device = torch.device('cpu') 
    print(f"Using local device: {device}")

    if not os.path.exists(model_path):
        print(f"ERROR: No model file found at {model_path}.")
        return

    # 1. Load DCUNet Model
    model = DeepComplexUNet(n_channels=1)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded successfully.")

    # 2. Audio settings
    target_sr = 16000
    n_fft = 512
    hop_length = 256
    
    # LOAD using soundfile instead of torchaudio
    if not os.path.exists(noisy_wav_path):
        print(f"ERROR: File {noisy_wav_path} not found.")
        return

    # sf.read returns (data, samplerate)
    audio_data, sr = sf.read(noisy_wav_path)
    
    # Convert to torch tensor [Channels, Time]
    # soundfile usually returns [Time, Channels]
    if len(audio_data.shape) > 1:
        # Stereo to Mono
        audio_data = np.mean(audio_data, axis=1)
    
    noisy_waveform = torch.FloatTensor(audio_data).unsqueeze(0)

    # Peak Normalization: Boost quiet audio to max volume before processing
    # (Matches training distribution)
    max_amp = torch.max(torch.abs(noisy_waveform)) + 1e-8
    noisy_waveform = noisy_waveform / max_amp

    # Resample if needed
    if sr != target_sr:
        noisy_waveform = torchaudio.transforms.Resample(sr, target_sr)(noisy_waveform)
    
    # 3. Apply STFT 
    stft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=None, normalized=True 
    )
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft, hop_length=hop_length, normalized=True
    )

    complex_spectrogram = stft(noisy_waveform)
    normalize_factor = torch.max(torch.abs(complex_spectrogram)) + 1e-8
    
    mix_real = (complex_spectrogram.real / normalize_factor).unsqueeze(0)
    mix_imag = (complex_spectrogram.imag / normalize_factor).unsqueeze(0)
    
    # 4. Predict Complex Mask
    with torch.no_grad():
        mask_real, mask_imag = model(mix_real, mix_imag)
        
    # Apply Complex Ratio Mask to the mixture
    pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
    pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real

    
    # Denormalize
    pred_clean_real = pred_clean_real.squeeze(0) * normalize_factor
    pred_clean_imag = pred_clean_imag.squeeze(0) * normalize_factor
    
    # 5. Re-combine 
    cleaned_complex = torch.complex(pred_clean_real, pred_clean_imag)
    
    # 6. Apply Inverse STFT
    cleaned_waveform = istft(cleaned_complex).squeeze(0).numpy()

    # Restore Original Volume
    cleaned_waveform = cleaned_waveform * max_amp.numpy()

    # SAVE using soundfile instead of torchaudio
    print(f"Saving DCUNet cleaned audio to {output_path}")
    sf.write(output_path, cleaned_waveform, target_sr)

if __name__ == "__main__":
    FINAL_MODEL_FILE = "complex_checkpoints/dcunet_epoch_120.pth"
    infer_local_audio(
        noisy_wav_path="test_audio/input.wav", 
        model_path=FINAL_MODEL_FILE, 
        output_path="test_audio_DNS_120th_result/input-clean.wav"
    )
