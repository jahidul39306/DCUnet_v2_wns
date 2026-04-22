import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

from pystoi import stoi
from complex_model import DeepComplexUNet

N_FFT = 512
HOP_LENGTH = 256
TARGET_SR = 16000
TEST_NOISY_DIR = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/voicebank_wav/test/noisy"
TEST_CLEAN_DIR = "/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4/voicebank_wav/test/clean"
OUTPUT_CSV = "evaluation_results.csv"

def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Match the n_channels=1 from training
    model = DeepComplexUNet().to(device) 
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def evaluate():
    # Priority: Latest, then 120, then 70
    model_path = "./complex_checkpoints/dcunet_epoch_60.pth"
    if not os.path.exists(model_path):
        model_path = "./complex_checkpoints/dcunet_epoch_120.pth"
    model, device = load_model(model_path)
    print(f"📡 Evaluating: {model_path} with N_FFT={N_FFT}")
    files = [f for f in os.listdir(TEST_NOISY_DIR) if f.endswith(".wav")]
    results = []
    window = torch.hann_window(N_FFT).to(device)

    print(f"Processing {len(files)} test files...")
    for f_name in tqdm(files):
        # 1. Load Audio (Bypass torchaudio)
        noisy_np, sr = sf.read(os.path.join(TEST_NOISY_DIR, f_name))
        clean_np, _ = sf.read(os.path.join(TEST_CLEAN_DIR, f_name))
        
        # Stereo to Mono if needed
        if noisy_np.ndim > 1: noisy_np = noisy_np.mean(axis=1)
        if clean_np.ndim > 1: clean_np = clean_np.mean(axis=1)

        noisy_wav = torch.from_numpy(noisy_np).float().to(device)
        if noisy_wav.ndim == 1: noisy_wav = noisy_wav.unsqueeze(0)
        # --- IMPROVEMENT: Normalisation ---
        max_amp = torch.max(torch.abs(noisy_wav)) + 1e-8
        noisy_wav_norm = noisy_wav / max_amp
        
        # 2. STFT
        stft = torch.stft(noisy_wav_norm, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True, normalized=True)
        real, imag = stft.real, stft.imag
        
        # Spectrogram Normalization
        norm_factor = torch.max(torch.abs(stft)) + 1e-8
        real_in = (real / norm_factor).unsqueeze(1) # (1, 1, Freq, Time)
        imag_in = (imag / norm_factor).unsqueeze(1)
        
        # 3. Enhance
        with torch.no_grad():
            mask_real, mask_imag = model(real_in, imag_in)
            
            # Complex Ratio Masking
            enhanced_real = real_in * mask_real - imag_in * mask_imag
            enhanced_imag = real_in * mask_imag + imag_in * mask_real
            
            # Denormalise and iSTFT
            enhanced_real = enhanced_real.squeeze(1) * norm_factor
            enhanced_imag = enhanced_imag.squeeze(1) * norm_factor
            
            enhanced_stft = torch.complex(enhanced_real, enhanced_imag)
            enhanced = torch.istft(enhanced_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, normalized=True)
            
            # Restore Volume
            enhanced = (enhanced * max_amp).squeeze().cpu().numpy()
        
        # 4. Results Alignment
        min_len = min(len(clean_np), len(enhanced))
        clean_eval = clean_np[:min_len]
        enhanced_eval = enhanced[:min_len]
        
        res = {"file": f_name}
        res["stoi"] = stoi(clean_eval, enhanced_eval, TARGET_SR, extended=False)
        res["sdr"] = 10 * np.log10(np.sum(clean_eval**2) / (np.sum((clean_eval - enhanced_eval)**2) + 1e-8))
        
        if HAS_PESQ:
            try:
                res["pesq"] = pesq(TARGET_SR, clean_eval, enhanced_eval, 'wb')
            except:
                res["pesq"] = None
        
        results.append(res)

    # 5. Save and Report
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*30)
    print("The final resut")
    print("="*30)
    print(f"Avg STOI:  {df['stoi'].mean():.4f}")
    print(f"Avg SDR:   {df['sdr'].mean():.2f} dB")
    if HAS_PESQ and "pesq" in df:
        print(f"Avg PESQ:  {df['pesq'].mean():.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
