import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS'] = '1'
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

def calculate_ssnr(clean, enhanced, frame_size=512):
    eps = 1e-10
    num_frames = len(clean) // frame_size
    if num_frames == 0: return 0
    
    snr_list = []
    for i in range(num_frames):
        c_seg = clean[i*frame_size : (i+1)*frame_size]
        e_seg = enhanced[i*frame_size : (i+1)*frame_size]
        
        clean_energy = np.sum(c_seg**2) + eps
        noise_energy = np.sum((c_seg - e_seg)**2) + eps
        
        snr = 10 * np.log10(clean_energy / noise_energy)
        snr = np.clip(snr, -10, 35) 
        snr_list.append(snr)
        
    return np.mean(snr_list)

N_FFT = 512
HOP_LENGTH = 256
TARGET_SR = 16000
TEST_NOISY_DIR = "./voicebank_wav/noisy"
TEST_CLEAN_DIR = "./voicebank_wav/clean"
OUTPUT_CSV = "evaluation_results.csv"

def load_model(path):
    device = torch.device('cpu')
    model = DeepComplexUNet().to(device) 
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def evaluate():
    model_path = "./complex_checkpoints/dcunet_epoch_120.pth"
    if not os.path.exists(model_path):
        model_path = "./complex_checkpoints/dcunet_epoch_60.pth"
    
    if not os.path.exists(model_path):
        print(f" ERROR: Could not find model at {model_path}!")
        return

    model, device = load_model(model_path)
    model = model.to(device)
    print(f"Evaluating (CPU Mode): {model_path}")
    
    files = [f for f in os.listdir(TEST_NOISY_DIR) if f.endswith(".wav")]
    results = []
    window = torch.hann_window(N_FFT).to(device)

    print(f"Processing {len(files)} test files")
    for f_name in files: # Removing tqdm temporarily to see clean debug prints
        print(f"Analyzing Sample: {f_name}")
        # 1. Load Audio (Bypass torchaudio)
        noisy_np, sr = sf.read(os.path.join(TEST_NOISY_DIR, f_name))
        clean_np, _ = sf.read(os.path.join(TEST_CLEAN_DIR, f_name))
        
        # Stereo to Mono if needed
        if noisy_np.ndim > 1: noisy_np = noisy_np.mean(axis=1)
        if clean_np.ndim > 1: clean_np = clean_np.mean(axis=1)

        noisy_wav = torch.from_numpy(noisy_np).float().to(device)
        if noisy_wav.ndim == 1: noisy_wav = noisy_wav.unsqueeze(0)
        
        # ---  Normalisation ---
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
        
        #  Minimum length check for STOI/PESQ 
        # STOI/PESQ require at least a few hundred milliseconds of audio
        if min_len < (TARGET_SR * 0.4): # 0.4 seconds
            print(f"Skipping {f_name}: Audio too short ({min_len} samples)")
            continue
            
        clean_eval = clean_np[:min_len]
        enhanced_eval = enhanced[:min_len]
        
        # --- NEW: NaN/Inf Cleaning ---
        enhanced_eval = np.nan_to_num(enhanced_eval, nan=0.0, posinf=0.0, neginf=0.0)
        
        res = {"file": f_name}
        print(f" Stats: Max Amp={np.max(np.abs(enhanced_eval)):.4f}, Length={len(enhanced_eval)}")
        print(f" Calculating STOI")
        res["stoi"] = stoi(clean_eval, enhanced_eval, TARGET_SR, extended=False)
        
        print(f"  Calculating SDR")
        res["sdr"] = 10 * np.log10(np.sum(clean_eval**2) / (np.sum((clean_eval - enhanced_eval)**2) + 1e-8))
        
        print(f"  Calculating SSNR")
        res["ssnr"] = calculate_ssnr(clean_eval, enhanced_eval)
        
        if HAS_PESQ:
            print(f" Calculating PESQ (Dangerous Step)")
            # Note: PESQ causes SegFaults if audio is too short or SR mismatch
            try:
                # Force double check of sample rate
                res["pesq"] = pesq(16000, clean_eval, enhanced_eval, 'wb')
                print(f"    PESQ Done.")
            except Exception as e:
                print(f"    PESQ Failed: {e}")
                res["pesq"] = None
        
        results.append(res)

    # 5. Save and Report
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*30)
    print(" The finsl result")
    print("="*30)
    print(f" Avg STOI:  {df['stoi'].mean():.4f}")
    print(f"Avg SDR:   {df['sdr'].mean():.2f} dB")
    print(f"Avg SSNR:  {df['ssnr'].mean():.2f} dB")
    if HAS_PESQ and "pesq" in df:
        print(f" Avg PESQ:  {df['pesq'].mean():.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
