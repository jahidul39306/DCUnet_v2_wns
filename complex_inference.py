import torch
import torchaudio
import soundfile as sf
import os
import pyloudnorm as pyln
import numpy as np
import pandas as pd
from complex_model import DeepComplexUNet
from pesq import pesq
from pystoi import stoi

TARGET_SR = 16000
N_FFT = 512
HOP_LENGTH = 256
COMPUTE_EVALUATION = True
OUTPUT_CSV = None

def infer_complex_audio(noisy_wav_path, model_path, output_normalized_noisy, file_name, output_normalized_enhanced="complex_cleaned_result.wav", clean_path = None, results = None ):
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
    
    if sr != TARGET_SR:
        noisy_waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(noisy_waveform)
    
    if noisy_waveform.shape[0] > 1:
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

    # Waveform Peak Normalization (to match training distribution)
    max_amp = torch.max(torch.abs(noisy_waveform)) + 1e-8
    noisy_waveform = noisy_waveform / max_amp

    # 3. Apply STFT 
    stft = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        power=None, 
        normalized=True # Must match complex_data_prep!
    )
    
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
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
    os.makedirs(os.path.dirname(output_enhanced_path), exist_ok=True) if os.path.dirname(output_enhanced_path) else None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_normalized_noisy), exist_ok=True) if os.path.dirname(output_normalized_noisy) else None

    print(f"Saving DCUNet cleaned audio to {output_enhanced_path}")
    
    noisy_waveform = normalize_loudness(noisy_waveform.squeeze(0).cpu().numpy(), TARGET_SR)
    cleaned_waveform = normalize_loudness(cleaned_waveform.squeeze(0).cpu().numpy(), TARGET_SR)
    
    if (COMPUTE_EVALUATION):
        if clean_path is not None:
            clean_np, _ = sf.read(clean_path)
            
            # Stereo to Mono if needed
            if clean_np.ndim > 1: 
                clean_np = clean_np.mean(axis=1)
                
            evaluate_file(file_name, clean_np, cleaned_waveform, noisy_waveform, results)
        else:
            Exception("Clean path must be defined for evaluation")
        
        
    
    sf.write(output_normalized_enhanced, cleaned_waveform, TARGET_SR)
    sf.write(output_normalized_noisy, noisy_waveform, TARGET_SR)
    
    
def normalize_loudness(audio, sample_rate, target_lufs=-23.0):
   
    meter = pyln.Meter(sample_rate)
    current_loudness = meter.integrated_loudness(audio)
    
    print(f"Current Loudness: {current_loudness:.2f} LUFS")
    
    normalized_audio = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    
    max_peak = np.max(np.abs(normalized_audio))
    if max_peak >= 1.0:
        normalized_audio = normalized_audio / (max_peak + 1e-6)
        
    return normalized_audio

def evaluate_file(f_name, clean_np, enhanced_np, noisy_np, results):
    min_len = min(len(clean_np), len(enhanced_np), len(noisy_np))
    clean_eval = clean_np[:min_len]
    enhanced_eval = enhanced_np[:min_len]
    noisy_eval = noisy_np[:min_len]
   
    res = {"file": f_name}
    res["stoi_en"] = stoi(clean_eval, enhanced_eval, TARGET_SR, extended=False)
    res["stoi_no"] = stoi(clean_eval, noisy_eval, TARGET_SR, extended=False)
    res["sdr_en"] = 10 * np.log10(np.sum(clean_eval**2) / (np.sum((clean_eval - enhanced_eval)**2) + 1e-8))
    res["sdr_no"] = 10 * np.log10(np.sum(clean_eval**2) / (np.sum((clean_eval - noisy_eval)**2) + 1e-8))
    res["si-sdr-en"] = compute_si_sdr(clean_eval, enhanced_eval)
    res["si-sdr-no"] = compute_si_sdr(clean_eval, noisy_eval)
    
    res["pesq_en"] = pesq(TARGET_SR, clean_eval, enhanced_eval, 'wb')
    res["pesq_no"] = pesq(TARGET_SR, clean_eval, noisy_eval , 'wb')
    
    results.append(res)
    
def print_and_save_evaluation_results(results):
    df = pd.DataFrame(results)
    if OUTPUT_CSV is not None:
       df.to_csv(OUTPUT_CSV, index=False) 
    else:
        Warning("CSV path is not defined. Results have not been saved")
    
    print("\n" + "=" * 50)
    print(f"{'Metric':<10} {'Enhanced':<15} {'Noisy':<15}")
    print("=" * 50)

    print(f"{'STOI':<10} "
        f"{df['stoi_en'].mean():<15.4f} "
        f"{df['stoi_no'].mean():<15.4f}")

    print(f"{'SDR (dB)':<10} "
        f"{df['sdr_en'].mean():<15.2f}"
        f"{df['sdr_no'].mean():<15.2f}")

    print(f"{'SI-SDR(dB)':<10} "
        f"{df['si-sdr-en'].mean():<15.2f}"
        f"{df['si-sdr-no'].mean():<15.2f}")

    print(f"{'PESQ':<10} "
        f"{df['pesq_en'].mean():<15.4f} "
        f"{df['pesq_no'].mean():<15.4f}")

    print("=" * 50)

    
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
    

if __name__ == "__main__":
    
    #MODEL_WEIGHTS = "C:/Users/zikan/Uni/erasmus2026/PBLproject/jahid_code_version/DCUnet_v2_wns/complex_checkpoints/model_with_overlap/dcunet_epoch_100.pth"
    #INPUT_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-1st/test_sets/dynamic_shield/25cm/segmented"
    #MODEL_WEIGHTS = "C:/Users/zikan/Uni/erasmus2026/PBLproject/jahid_code_version/DCUnet_v2_wns/complex_checkpoints/model_no_overlap/best_model_4.pth"
    MODEL_WEIGHTS = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/TESTING/OVERLAP/dcunet_epoch_overlap.pth"
    INPUT_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_smart/loudness-test/noisy"
    #INPUT_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/TESTING/testing_spectral_leakage"
    #OUTPUT_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/TESTING/OVERLAP" # Saving back into the same folder for convenience
    #OUTPUT_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/TESTING/testing_spectral_leakage"
    OUTPUT_ENHANCED =  "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_smart/loudness-test/clean"
    NOISY_NORMALIZED = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_smart/loudness-test/normalized-noisy"
    CLEAN_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_smart/loudness-test/clean"
    COMPUTE_EVALUATION = True
    
    if COMPUTE_EVALUATION: 
        results_list = []
    
    if not os.path.exists(MODEL_WEIGHTS):
        MODEL_WEIGHTS = "./complex_checkpoints/dcunet_epoch_120.pth"

    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav") and not f.startswith("cleaned_")]
    
    if not files:
        print(f"No .wav files found in {INPUT_DIR}")
    else:
        print(f"Cleaning {len(files)} files...")

        for file_name in files:
            input_path = os.path.join(INPUT_DIR, file_name)
            print(input_path)
            output_name = f"{os.path.splitext(file_name)[0]}.wav"
            output_enhanced_path = os.path.join(OUTPUT_ENHANCED, output_name)
            noisy_normalized_path = os.path.join(NOISY_NORMALIZED, output_name)
            
            
            print(f"Processing: {file_name} -> {output_name}")
            if (COMPUTE_EVALUATION):
                clean_path = os.path.join(CLEAN_DIR, file_name)
                print(clean_path)
                print("COMPUTE EVALUATION")
                infer_complex_audio(
                    noisy_wav_path=input_path, 
                    model_path=MODEL_WEIGHTS, 
                    output_normalized_noisy=noisy_normalized_path,
                    file_name = file_name,
                    output_normalized_enhanced=output_enhanced_path,
                    clean_path = clean_path,
                    results = results_list
                )
            else:
                infer_complex_audio(
                    noisy_wav_path=input_path, 
                    model_path=MODEL_WEIGHTS, 
                    output_normalized_noisy=noisy_normalized_path,
                    file_name = file_name,
                    output_normalized_enhanced=output_enhanced_path,
                )
        
        print("\nAll files processed successfully!")
        if (COMPUTE_EVALUATION):
            print_and_save_evaluation_results(results_list)
