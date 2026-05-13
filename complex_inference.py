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
from dnsmos_local import main_dnsmos
from torch_squim_evaluator import SquimEvaluator
from config import Config
import librosa
from scipy import signal
import matplotlib.pyplot as plt

N_FFT = 512
HOP_LENGTH = 256
OUTPUT_CSV = None



def infer_complex_audio(model, noisy_np, output_normalized_noisy, file_name, output_normalized_enhanced="complex_cleaned_result.wav", clean_path = None, results = None ):
    """Run inference. If COPMUTE_EVALUATION_METRICS is true than the evaluation metric is computed after inference.

    Args:
        noisy_wav_path: directory of the noisy files
        model_path: directory of the model
        output_normalized_noisy: direcotry where the normalized noisy files will be saved (loudness)
        file_name: name of the currently processed file
        output_normalized_enhanced: output enhanced file name. Defaults to "complex_cleaned_result.wav".
        clean_path: directory of the corresponding reference files (necessary for evaluation). Defaults to None.
        results: Array for the results of evaluation. Defaults to None.
    """

    # # Load and prep audio
    # if not os.path.exists(noisy_wav_path):
    #     print(f"Error: File not found: {noisy_wav_path}")
    #     return

    # data, sr = sf.read(noisy_wav_path, dtype='float32')
    if noisy_np.ndim == 1:
        noisy_waveform = torch.from_numpy(noisy_np).unsqueeze(0)
    else:
        # soundfile returns [Time, Channels], PyTorch needs [Channels, Time]
        noisy_waveform = torch.from_numpy(noisy_np.T)
    
    # if sr != TARGET_SR:
    #     noisy_waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(noisy_waveform)
    
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
    os.makedirs(os.path.dirname(output_normalized_enhanced), exist_ok=True) if os.path.dirname(output_normalized_enhanced) else None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_normalized_noisy), exist_ok=True) if os.path.dirname(output_normalized_noisy) else None

    print(f"Saving DCUNet cleaned audio to {output_normalized_enhanced}")
    
    return cleaned_waveform
    
    #noisy_waveform_np = noisy_waveform.squeeze(0).cpu().numpy()
    #cleaned_waveform_np = cleaned_waveform.squeeze(0).cpu().numpy()
    
    # if COMPUTE_EVALUATION_METRICS:
    #     print("Evaluating...")
    #     if clean_path is not None:
    #         clean_np, sr_ = sf.read(clean_path)
    #         if (sr_ != TARGET_SR):
    #             print("Resampling clean file..")
    #             clean_np = librosa.resample(clean_np, orig_sr=sr_, target_sr=TARGET_SR)
            
    #         # Normalize loudness of the clean file 
    #         gain_factor_cl = get_loudness_norm_factor(clean_np, TARGET_SR)
    #         clean_norm_np = clean_np * gain_factor_cl
            
    #         # Scale the other signals using the same factor (to keep the speech level consistent)
    #         noisy_norm_np = noisy_waveform_np * gain_factor_cl          
    #         enhanced_norm_np = cleaned_waveform_np * gain_factor_cl
            
    #         # If some peak exceeds maximum allowed value, normalize:
    #         max_val = max(np.max(np.abs(clean_norm_np)), np.max(np.abs(noisy_norm_np)), np.max(np.abs(enhanced_norm_np)))
    #         if max_val > 1.0:
    #             clean_norm_np /= max_val
    #             noisy_norm_np /= max_val
    #             enhanced_norm_np /= max_val
                
    #             # Stereo to Mono if needed
    #             if clean_np.ndim > 1: 
    #                 clean_np = clean_np.mean(axis=1)
            
    #         # Align the signals using cross correlation
    #         clean_norm_np, noisy_norm_np, enhanced_norm_np = align_signals(clean_norm_np, noisy_norm_np, enhanced_norm_np)
              
    #         # Compute evaluation metrics  
    #         evaluate_file(file_name, clean_norm_np, enhanced_norm_np, noisy_norm_np, noisy_waveform, cleaned_waveform, results)
            
    #     else:
    #         raise Exception("Clean path must be defined for evaluation")
    # else:
    #     # Normalize loudness (to match -23 LUFS)
    #     gain_factor_no = get_loudness_norm_factor(noisy_waveform_np, TARGET_SR)
    #     noisy_waveform_np = noisy_waveform_np * gain_factor_no
        
    #     gain_factor_en = get_loudness_norm_factor(cleaned_waveform_np, TARGET_SR)
    #     cleaned_waveform_np = cleaned_waveform_np * gain_factor_en

    # # Save the normalized files
    # sf.write(output_normalized_enhanced, cleaned_waveform_np, TARGET_SR)
    # sf.write(output_normalized_noisy, noisy_waveform_np, TARGET_SR)
    

    
# def perform_only_test(file_name, noisy_wav_path, clean_wav_path, enhanced_wav_path, results):
#     # performs only the test from saved files (does not do the inference)

#     noisy_np, sr_no = sf.read(noisy_wav_path, dtype='float32')
#     clean_np, sr_cl = sf.read(clean_wav_path, dtype='float32')
#     enhanced_np, sr_en = sf.read(enhanced_wav_path, dtype='float32')
    
#     if noisy_np.ndim == 1:
#         noisy_torch = torch.from_numpy(noisy_np).unsqueeze(0)
#     else:
#         # soundfile returns [Time, Channels], PyTorch needs [Channels, Time]
#         noisy_torch = torch.from_numpy(noisy_np.T)
        
#     if enhanced_np.ndim == 1:
#         enhanced_torch = torch.from_numpy(enhanced_np).unsqueeze(0)
#     else:
#         # soundfile returns [Time, Channels], PyTorch needs [Channels, Time]
#         enhanced_torch = torch.from_numpy(enhanced_np.T)
    
#     if not (sr_no == sr_cl == sr_en == TARGET_SR):
#         raise Exception("Sampling rates must match")
    
#     gain_factor_cl = get_loudness_norm_factor(clean_np, TARGET_SR)
    
#     clean_norm_np = clean_np * gain_factor_cl
    
#     noisy_norm_np  = noisy_np * gain_factor_cl
    
#     enhanced_norm_np = enhanced_np * gain_factor_cl
    
#     max_val = max(np.max(np.abs(clean_norm_np)), np.max(np.abs(noisy_norm_np)), np.max(np.abs(enhanced_norm_np)))
#     if max_val > 1.0:
#         clean_norm_np /= max_val
#         noisy_norm_np /= max_val
#         enhanced_norm_np /= max_val
        
#     clean_norm_np, noisy_norm_np, enhanced_norm_np = align_signals(clean_norm_np, noisy_norm_np, enhanced_norm_np)
    
#     evaluate_file(file_name, clean_norm_np, enhanced_norm_np, noisy_norm_np, noisy_torch, enhanced_torch, results)
    
    
def align_signals(clean, noisy, enhanced):
# time align signals using cross correlation

    # def apply_highpass(sig):
    #     # Use Second-Order Sections (sos) for better numerical stability
    #     # sos = signal.butter(4, cutoff, btype='highpass', fs=fs, output='sos')
    #     return signal.sosfiltfilt(hp_filter, sig)

    def estimate_delay(ref, sig):
        #ref_filt = apply_highpass(ref)
        #sig_filt = apply_highpass(sig)
        
        corr = np.correlate(sig, ref, mode='full')
        lags = np.arange(-len(ref) + 1, len(sig))
        delay = lags[np.argmax(corr)]

        return delay


    def shift(sig, delay, target_len):

        if delay > 0:
            # signal delayed
            sig = sig[delay:]

        elif delay < 0:
            # signal ahead
            sig = np.pad(sig, (abs(delay), 0))

        # match target length
        if len(sig) < target_len:
            sig = np.pad(sig, (0, target_len - len(sig)))
        else:
            sig = sig[:target_len]

        return sig

    # estimate delays
    noisy_delay = estimate_delay(clean, noisy)
    enhanced_delay = estimate_delay(clean, enhanced)

    # align
    target_len = len(clean)

    noisy_aligned = shift(noisy, noisy_delay, target_len)
    enhanced_aligned = shift(enhanced, enhanced_delay, target_len)
    clean_aligned = clean[:target_len]

    return (
        clean_aligned,
        noisy_aligned,
        enhanced_aligned,
    )

  
def get_loudness_norm_factor(audio, meter, sample_rate, target_lufs=-23.0):
   
    current_loudness = meter.integrated_loudness(audio)
    
    print(f"Current Loudness: {current_loudness:.2f} LUFS")
    
    gain_db = target_lufs - current_loudness
    gain_factor = 10**(gain_db / 20)
    
    return gain_factor

def evaluate_file(f_name, clean_np, enhanced_np, noisy_np, noisy_torch, enhanced_torch, squim_evaluator, loud_meter, results, target_sr):
    
    print("Evaluating...")
    
    # Normalize loudness of the clean file 
    gain_factor_cl = get_loudness_norm_factor(clean_np, loud_meter, target_sr)
    clean_norm_np = clean_np * gain_factor_cl
    
    # Scale the other signals using the same factor (to keep the speech level consistent)
    noisy_norm_np = noisy_np * gain_factor_cl          
    enhanced_norm_np = enhanced_np * gain_factor_cl
    
    # If some peak exceeds maximum allowed value, normalize:
    max_val = max(np.max(np.abs(clean_norm_np)), np.max(np.abs(noisy_norm_np)), np.max(np.abs(enhanced_norm_np)))
    if max_val > 1.0:
        clean_norm_np /= max_val
        noisy_norm_np /= max_val
        enhanced_norm_np /= max_val
    
    # Zero-mean normalization
    clean_norm_np = clean_norm_np - np.mean(clean_norm_np)
    enhanced_norm_np = enhanced_norm_np - np.mean(enhanced_norm_np)
    noisy_norm_np = noisy_norm_np - np.mean(noisy_norm_np)
    
    # Align the signals using cross correlation
    clean_norm_np, noisy_norm_np, enhanced_norm_np = align_signals(clean_norm_np, noisy_norm_np, enhanced_norm_np)
    
    # # Match length
    # min_len = min(len(clean_np), len(enhanced_np), len(noisy_np))
    # clean_eval = clean_np[:min_len]
    # enhanced_eval = enhanced_np[:min_len]
    # noisy_eval = noisy_np[:min_len]
    
    
    # Compute scores
    scores_noisy = compute_squim(noisy_torch, squim_evaluator)
    scores_enhanced = compute_squim(enhanced_torch, squim_evaluator)
   
    res = {"file": f_name}
    res["stoi_en"] = stoi(clean_norm_np, enhanced_norm_np, target_sr, extended=True)
    print(f"STOI en {res['stoi_en']}")
    res["stoi_no"] = stoi(clean_norm_np, noisy_norm_np, target_sr, extended=True)
    res["sdr_en"] = 10 * np.log10(np.sum(clean_norm_np**2) / (np.sum((clean_norm_np - enhanced_norm_np)**2) + 1e-8))
    res["sdr_no"] = 10 * np.log10(np.sum(clean_norm_np**2) / (np.sum((clean_norm_np - noisy_norm_np)**2) + 1e-8))
    res["si-sdr-en"] = compute_si_sdr(clean_norm_np, enhanced_norm_np)
    print(f"Si SDR en {res['si-sdr-en']}")
    res["si-sdr-no"] = compute_si_sdr(clean_norm_np, noisy_norm_np)
    
    res["squim-stoi-no"] = scores_noisy["stoi"]
    res["squim-pesq-no"] = scores_noisy["pesq"]
    res["squim-si-sdr-no"] = scores_noisy["si_sdr"]
    #res["squim-mos-no"] = scores_noisy["mos"]
    
    res["squim-stoi-en"] = scores_enhanced["stoi"]
    res["squim-pesq-en"] = scores_enhanced["pesq"]
    res["squim-si-sdr-en"] = scores_enhanced["si_sdr"]
    #res["squim-mos-en"] = scores_enhanced["mos"]
    
    try:
        res["pesq_en"] = pesq(target_sr, clean_norm_np, enhanced_norm_np, 'wb')
        print(res["pesq_en"] )
        res["pesq_no"] = pesq(target_sr, clean_norm_np, noisy_norm_np , 'wb')
    except:
        print("Warning: PESQ failed")
        res["pesq_en"] = None
        res["pesq_no"] = None
    
    results.append(res)

def print_and_save_evaluation_results(results):
    df = pd.DataFrame(results)
    if OUTPUT_CSV is not None:
       df.to_csv(OUTPUT_CSV, index=False) 
    else:
        print("CSV path is not defined. Results have not been saved")
    
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
    
    
    print(f"{'Squim STOI':<10} "
        f"{df['squim-stoi-en'].mean():<15.4f} "
        f"{df['squim-stoi-no'].mean():<15.4f}")
    
    print(f"{'Squim PESQ':<10} "
        f"{df['squim-pesq-en'].mean():<15.4f} "
        f"{df['squim-pesq-no'].mean():<15.4f}")
    
    print(f"{'Squim SI-SDR':<10} "
        f"{df['squim-si-sdr-en'].mean():<15.4f} "
        f"{df['squim-si-sdr-no'].mean():<15.4f}")

    print("=" * 50)

    
def compute_si_sdr(reference, enhanced, eps=1e-8):
    
    # Projection of estimation onto reference
    alpha = np.dot(enhanced, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference

    # Noise/error component
    noise = target - enhanced

    # SI-SDR
    ratio = (np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio)
 
def compute_squim(audio_torch, squim_evaluator):
    
    scores = squim_evaluator.evaluate(audio_torch)
    
    return scores

def load_audio(audio_path, target_sr):
    
    if not os.path.exists(audio_path):
        raise Exception(f"Error: File not found: {audio_path}")

    audio_np, sr = sf.read(audio_path, dtype='float32')
    
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    
    if sr != target_sr:
        librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        
    if audio_np.ndim == 1:
        audio_torch = torch.from_numpy(audio_np).unsqueeze(0)
    else:
        # soundfile returns [Time, Channels], PyTorch needs [Channels, Time]
        audio_torch = torch.from_numpy(audio_np.T)
    
    return audio_np, audio_torch
    
def model_loader(model_path, device):
    # Load DCUNet Model
    model = DeepComplexUNet(n_channels=1)

    print(f"Using device: {device}") 
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {MODEL_WEIGHTS}.")
    
    return model, checkpoint

def normalize_loudness(audio, meter, target_sr, target_lufs):
    gain_factor = get_loudness_norm_factor(audio, meter, target_sr, target_lufs)
    
    return gain_factor * audio


if __name__ == "__main__":
    
    # Directory where the model is saved
    MODEL_WEIGHTS = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/TESTING/NO_OVERLAP/best_model_5.pth"
    
    if not os.path.exists(MODEL_WEIGHTS):
        raise Exception("Directory for Model Weights does not exist")
    
    # Directory for noisy files
    INPUT_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_sequential/SNR_-5"
    #INPUT_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_sequential/output/snr_-5/no_overlap/analysis/sample_1/noisy"
   
    # Directory for saving enhanced files
    OUTPUT_ENHANCED =  "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_sequential/output/snr_-5/no_overlap/"
    
    # PATH FOR SAVING NOISY FILES AFTER lOUDNESS NORMALIZATION 
    NOISY_NORMALIZED = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_sequential/output/snr_-5/normalized_noisy"
     
    config = Config()
    
    if config.compute_metrics:
        # Path where the clean speech files are saved (necessary to compute intrusive metrics)
        CLEAN_DIR = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/test_sets/wind_plus_valentini_sequential/speech_norm/SNR_-5"
        eval_results = []
        squim_evaluator = SquimEvaluator()
        # hp_filter = signal.butter(4, 200, btype='highpass', fs = config.target_sr, output='sos')
    
    loud_meter = pyln.Meter(config.target_sr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, checkpoint = model_loader(MODEL_WEIGHTS, device)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
    
    if not files:
        print(f"No .wav files found in {INPUT_DIR}")
    else:
        print(f"Cleaning {len(files)} files...")
        
        for file_name in files:
            
            input_path = os.path.join(INPUT_DIR, file_name)
            
            output_name = f"{os.path.splitext(file_name)[0]}.wav"
            
            output_enhanced_path = os.path.join(OUTPUT_ENHANCED, output_name)
            
            noisy_normalized_path = os.path.join(NOISY_NORMALIZED, output_name)
            
            
            noisy_np, noisy_torch = load_audio(input_path, config.target_sr)
            
            
            print(f"Processing: {file_name} -> {output_name}")
            
            enhanced_torch = infer_complex_audio(
                    model=model,
                    noisy_np=noisy_np, 
                    output_normalized_noisy=noisy_normalized_path,
                    file_name = file_name,
                    output_normalized_enhanced=output_enhanced_path,
            )
            
            enhanced_np = enhanced_torch.squeeze(0).cpu().numpy()
            
            if (config.compute_metrics):
                clean_wav_path = os.path.join(CLEAN_DIR, file_name)
                
                clean_np, _ = load_audio(clean_wav_path, config.target_sr)
                
                print("REFERENCE SPEECH PATH: "+clean_wav_path)
                
                evaluate_file(
                    f_name=file_name,
                    clean_np=clean_np,
                    enhanced_np=enhanced_np,
                    noisy_np=noisy_np,
                    noisy_torch=noisy_torch,
                    enhanced_torch=enhanced_torch,
                    squim_evaluator=squim_evaluator,
                    loud_meter=loud_meter,
                    results=eval_results,
                    target_sr=config.target_sr
                )
                
                
            enhanced_np = normalize_loudness(enhanced_np, loud_meter, config.target_sr, config.target_lufs)
            noisy_np =  normalize_loudness(noisy_np, loud_meter, config.target_sr, config.target_lufs)
            
            sf.write(output_enhanced_path, enhanced_np, config.target_sr)
            sf.write(noisy_normalized_path, noisy_np, config.target_sr)
            
        print("\nAll files processed successfully!")  
            
        if (config.compute_metrics):   
            print_and_save_evaluation_results(eval_results)
            
        
                
                # if (PERFORM_ONLY_TEST):
                # # if inference was done earlier and only tests must be performed
                    
                #     enhanced_wav_path = os.path.join(OUTPUT_ENHANCED, file_name)
                #     perform_only_test(file_name, input_path, clean_wav_path, enhanced_wav_path, results_list)
                # else:
                # do inference and tests
                    # infer_complex_audio(
                    #     model=model,
                    #     checkpoint=checkpoint,
                    #     noisy_wav_path=input_path,  
                    #     output_normalized_noisy=noisy_normalized_path,
                    #     file_name = file_name,
                    #     output_normalized_enhanced=output_enhanced_path,
                    #     clean_path = clean_wav_path,
                    #     results = results_list
                    # )
            # else:
            # # do only inference (no testing)
            #     infer_complex_audio(
            #         model=model,
            #         checkpoint=checkpoint,
            #         noisy_wav_path=input_path, 
            #         output_normalized_noisy=noisy_normalized_path,
            #         file_name = file_name,
            #         output_normalized_enhanced=output_enhanced_path,
            #     )
            
        
        
        
        # if (COMPUTE_EVALUATION_METRICS):
        #     print("MODEL "+MODEL_WEIGHTS)
        #     print("INPUT (NOISY) "+INPUT_DIR)
            
        #     # PRINT RESULTS of the tests
        #     print_and_save_evaluation_results(results_list)
            
        #     # DNSMOS TEST 
        #     print("DNSMOS enhanced:")
        #     main_dnsmos(OUTPUT_ENHANCED)
        #     print("DNSMOS noisy:")
        #     main_dnsmos(NOISY_NORMALIZED)
            
            