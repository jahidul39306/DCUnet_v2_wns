import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import pyloudnorm as pyln


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
TEST_NOISY_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_smart/SNR_-5"
TEST_CLEAN_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_smart/speech"
OUTPUT_CSV = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_smart/evaluation_results_-5.csv"

def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Match the n_channels=1 from training
    model = DeepComplexUNet().to(device) 
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def compute_si_sdr(reference, enhanced, eps=1e-8):
    
    # Zero-mean normalization
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)

    # Projection of estimation onto reference
    alpha = np.dot(enhanced, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference

    # Noise/error component
    noise = target - enhanced

    # SI-SDR
    ratio = (np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio)

def normalize_loudness(audio, sample_rate, target_lufs=-23.0):
   
    meter = pyln.Meter(sample_rate)
    current_loudness = meter.integrated_loudness(audio)
    
    print(f"Current Loudness: {current_loudness:.2f} LUFS")
    
    normalized_audio = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    
    max_peak = np.max(np.abs(normalized_audio))
    if max_peak >= 1.0:
        normalized_audio = normalized_audio / (max_peak + 1e-6)
        
    return normalized_audio

def evaluate():
    # Priority: Latest, then 120, then 70
    model_path = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/TESTING/OVERLAP/dcunet_epoch_overlap.pth"
    #if not os.path.exists(model_path):
    #    model_path = "./complex_checkpoints/dcunet_epoch_120.pth"
    model, device = load_model(model_path)
    print(f"Evaluating: {model_path} with N_FFT={N_FFT}")
    files = [f for f in os.listdir(TEST_NOISY_DIR) if f.endswith(".wav")]
    results = []
    window = torch.hann_window(N_FFT).to(device)
    
    print(f"Processing {len(files)} test files...")
    for f_name in tqdm(files):
        # 1. Load Audio (Bypass torchaudio)
        
        #clean_f_name = f_name.replace("cleaned_mixed_th_", "") 
        #clean_f_name = f_name.removeprefix("cleaned_mixed_th_").removesuffix("_120")
        
        noisy_np, sr = sf.read(os.path.join(TEST_NOISY_DIR, f_name))
        clean_np, _ = sf.read(os.path.join(TEST_CLEAN_DIR, f_name))
        
        # Stereo to Mono if needed
        if noisy_np.ndim > 1: 
            noisy_np = noisy_np.mean(axis=1)
        if clean_np.ndim > 1: 
            clean_np = clean_np.mean(axis=1)

        noisy_wav = torch.from_numpy(noisy_np).float().to(device)
        if noisy_wav.ndim == 1: 
            noisy_wav = noisy_wav.unsqueeze(0)
        
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
        
        #enhanced = normalize_loudness(enhanced, TARGET_SR, target_lufs=-23.0)
        #clean_np = normalize_loudness(clean_np, TARGET_SR, target_lufs=-23.0)
        #noisy_np = normalize_loudness(noisy_np, TARGET_SR, target_lufs=-23.0)
        
        # Results Alignment
        min_len = min(len(clean_np), len(enhanced), len(noisy_np))
        clean_eval = clean_np[:min_len]
        enhanced_eval = enhanced[:min_len]
        noisy_eval = noisy_np[:min_len]
        
        noisy_eval = noisy_eval / max(abs(noisy_eval))
        clean_eval = clean_eval / max(abs(clean_eval))
        enhanced_eval = enhanced_eval / max(abs(enhanced_eval))
        
        res = {"file": f_name}
        res["stoi_en"] = stoi(clean_eval, enhanced_eval, TARGET_SR, extended=False)
        res["stoi_no"] = stoi(clean_eval, noisy_eval, TARGET_SR, extended=False)
        res["sdr_en"] = 10 * np.log10(np.sum(clean_eval**2) / (np.sum((clean_eval - enhanced_eval)**2) + 1e-8))
        res["sdr_no"] = 10 * np.log10(np.sum(clean_eval**2) / (np.sum((clean_eval - noisy_eval)**2) + 1e-8))
        res["si-sdr-en"] = compute_si_sdr(clean_eval, enhanced_eval)
        res["si-sdr-no"] = compute_si_sdr(clean_eval, noisy_eval)
        
        #clean_eval = clean_eval / max(clean_eval)
        
        
        if HAS_PESQ:
            try:
                res["pesq_en"] = pesq(TARGET_SR, clean_eval, enhanced_eval, 'wb')
                res["pesq_no"] = pesq(TARGET_SR, clean_eval, noisy_eval , 'wb')
                
            except:
                res["pesq_en"] = None
                res["pesq_no"] = None
        
        results.append(res)

    # 5. Save and Report
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*30)
    print("The final resut")
    print("="*30)
    print(f"Avg STOI enhanced:  {df['stoi_en'].mean():.4f}")
    print(f"Avg STOI noisy:  {df['stoi_no'].mean():.4f}")
    print(f"Avg SDR enhanced:   {df['sdr_en'].mean():.2f} dB")
    print(f"Avg SDR noisy:   {df['sdr_no'].mean():.2f} dB")
    print(f"Avg SI-SDR enhanced:   {df['si-sdr-en'].mean():.2f} dB")
    print(f"Avg SI-SDR noisy:   {df['si-sdr-no'].mean():.2f} dB")
    if HAS_PESQ:
        print(f"Avg PESQ enhanced:  {df['pesq_en'].mean():.4f}")
        print(f"Avg PESQ noisy:  {df['pesq_no'].mean():.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
