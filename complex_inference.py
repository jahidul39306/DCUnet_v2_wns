import torch
import torchaudio
import soundfile as sf
import os
from complex_model import DeepComplexUNet

def infer_complex_audio(noisy_wav_path, model_path, output_path="complex_cleaned_result.wav"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load DCUNet Model
    model = DeepComplexUNet(n_channels=1)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}.")

    # 2. Audio settings
    sample_rate = 16000
    n_fft = 512
    hop_length = 256
    
    # Load and prep audio
    if not os.path.exists(noisy_wav_path):
        print(f"Error: File not found: {noisy_wav_path}")
        return

    data, sr = sf.read(noisy_wav_path, dtype='float32')
    if data.ndim == 1:
        noisy_waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        # soundfile returns [Time, Channels], PyTorch needs [Channels, Time]
        noisy_waveform = torch.from_numpy(data.T)
    
    if sr != sample_rate:
        noisy_waveform = torchaudio.transforms.Resample(sr, sample_rate)(noisy_waveform)
    
    if noisy_waveform.shape[0] > 1:
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

    # Waveform Peak Normalization (to match training distribution)
    max_amp = torch.max(torch.abs(noisy_waveform)) + 1e-8
    noisy_waveform = noisy_waveform / max_amp

    # 3. Apply STFT 
    stft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, 
        hop_length=hop_length, 
        power=None, 
        normalized=True # Must match complex_data_prep!
    )
    
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft, 
        hop_length=hop_length, 
        normalized=True
    )

    complex_spectrogram = stft(noisy_waveform)
    
    eps = 1e-8
    normalize_factor = torch.max(torch.abs(complex_spectrogram)) + eps
    
    # Isolate Real and Imaginary and Normalize
    mix_real = complex_spectrogram.real / normalize_factor
    mix_imag = complex_spectrogram.imag / normalize_factor
    
    # Add batch & channel dims [Batch, Channels, Freq, Time]
    mix_real_input = mix_real.unsqueeze(0).to(device)
    mix_imag_input = mix_imag.unsqueeze(0).to(device)
    
    # 4. Predict Complex Mask
    with torch.no_grad():
        mask_real, mask_imag = model(mix_real_input, mix_imag_input)
        
    # Apply Complex Ratio Mask to the mixture
    pred_clean_real = mask_real * mix_real_input - mask_imag * mix_imag_input
    pred_clean_imag = mask_real * mix_imag_input + mask_imag * mix_real_input

    
    # Denormalize
    pred_clean_real = pred_clean_real.squeeze(0) * normalize_factor
    pred_clean_imag = pred_clean_imag.squeeze(0) * normalize_factor
    
    # 5. Re-combine into a raw Complex Tensor
    # We no longer need Griffin-Lim because the model PREDICTED the phase geometry for us!
    cleaned_complex = torch.complex(pred_clean_real, pred_clean_imag).cpu()
    
    # 6. Apply Inverse STFT
    cleaned_waveform = istft(cleaned_complex)
    
    # Restore Original Volume
    cleaned_waveform = cleaned_waveform * max_amp

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    print(f"Saving DCUNet cleaned audio to {output_path}")
    sf.write(output_path, cleaned_waveform.squeeze(0).cpu().numpy(), sample_rate)

if __name__ == "__main__":
    MODEL_WEIGHTS = "./complex_checkpoints/latest_checkpoint.pth"
    INPUT_DIR = "test123"
    OUTPUT_DIR = "test123" # Saving back into the same folder for convenience

    if not os.path.exists(MODEL_WEIGHTS):
        MODEL_WEIGHTS = "./complex_checkpoints/dcunet_epoch_120.pth"

    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav") and not f.startswith("cleaned_")]
    
    if not files:
        print(f"No .wav files found in {INPUT_DIR}")
    else:
        print(f"Cleaning {len(files)} files...")

        for file_name in files:
            input_path = os.path.join(INPUT_DIR, file_name)
            output_name = f"cleaned_{os.path.splitext(file_name)[0]}_120.wav"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            print(f"Processing: {file_name} -> {output_name}")
            infer_complex_audio(
               noisy_wav_path=input_path, 
               model_path=MODEL_WEIGHTS, 
               output_path=output_path
            )

        print("\nAll files processed successfully!")
