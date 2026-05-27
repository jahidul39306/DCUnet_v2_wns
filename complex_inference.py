import torch
import torchaudio
import soundfile as sf
import os
import numpy as np
import pandas as pd
from complex_model import DeepComplexUNet
from pesq import pesq
from pystoi import stoi
from torch_squim_evaluator import SquimEvaluator
from config import Config
import librosa
from scipy.signal import butter, sosfiltfilt
from convert_to_csv import save_to_csv


N_FFT = 512
HOP_LENGTH = 256

def compute_average_metrics(eval_results):
    if not eval_results:
        return {}
    keys = [k for k, v in eval_results[0].items() if isinstance(v, (int, float)) or v is None]
    averages = {}
    for k in keys:
        values = [r[k] for r in eval_results if r.get(k) is not None]
        averages[k] = sum(values) / len(values) if values else None
    return averages

def infer_complex_audio(model, noisy_np, output_normalized_noisy, output_normalized_enhanced="complex_cleaned_result.wav"):
    
    if noisy_np.ndim == 1:
        noisy_waveform = torch.from_numpy(noisy_np).unsqueeze(0)
    else:
        noisy_waveform = torch.from_numpy(noisy_np.T)
    
    
    if noisy_waveform.shape[0] > 1:
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

    max_amp = torch.max(torch.abs(noisy_waveform)) + 1e-8
    noisy_waveform = noisy_waveform / max_amp

    # Apply STFT 
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
    
    # Add batch and channel dims [Batch, Channels, Freq, Time]
    mix_real_input = mix_real.unsqueeze(0).to(device)
    mix_imag_input = mix_imag.unsqueeze(0).to(device)
    
    # Predict complex mask
    with torch.no_grad():
        mask_real, mask_imag = model(mix_real_input, mix_imag_input)
        
    # Apply the mask to the mixture
    pred_clean_real = mask_real * mix_real_input - mask_imag * mix_imag_input
    pred_clean_imag = mask_real * mix_imag_input + mask_imag * mix_real_input

    # Denormalize
    pred_clean_real = pred_clean_real.squeeze(0) * normalize_factor
    pred_clean_imag = pred_clean_imag.squeeze(0) * normalize_factor
    
    cleaned_complex = torch.complex(pred_clean_real, pred_clean_imag).cpu()
    
    # Apply iSTFT
    cleaned_waveform = istft(cleaned_complex)

    cleaned_waveform = cleaned_waveform * max_amp

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_normalized_enhanced), exist_ok=True) if os.path.dirname(output_normalized_enhanced) else None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_normalized_noisy), exist_ok=True) if os.path.dirname(output_normalized_noisy) else None

    print(f"Saving DCUNet cleaned audio to {output_normalized_enhanced}")
    
    return cleaned_waveform
  
def align_signals(clean, noisy, enhanced):
# time align signals using cross correlation
    def estimate_delay(ref, sig):      
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

    return (clean_aligned, noisy_aligned, enhanced_aligned,)

  
def get_loudness_norm_factor(audio, meter, sample_rate, target_lufs=-23.0):
   
    current_loudness = meter.integrated_loudness(audio)
    
    print(f"Current Loudness: {current_loudness:.2f} LUFS")
    
    gain_db = target_lufs - current_loudness
    gain_factor = 10**(gain_db / 20)
    
    return gain_factor

def evaluate_file(f_name, enhanced_np, noisy_np, noisy_torch, enhanced_torch, squim_evaluator, results, target_sr, clean_np, gain_factor):
    
    print("Evaluating...")
    
    if config.compute_intrusive_metrics:
        clean_norm_np = clean_np
        clean_norm_np = clean_norm_np - np.mean(clean_norm_np) 
    
    noisy_norm_np = noisy_np * gain_factor
    enhanced_norm_np = enhanced_np
    
    enhanced_norm_np = enhanced_norm_np - np.mean(enhanced_norm_np)
    noisy_norm_np = noisy_norm_np - np.mean(noisy_norm_np)
    
    # Align the signals using cross correlation
    if config.compute_intrusive_metrics:
        clean_norm_np, noisy_norm_np, enhanced_norm_np = align_signals(clean_norm_np, noisy_norm_np, enhanced_norm_np)
   
    res = {"file": f_name}
    
    if config.compute_intrusive_metrics:
        res["stoi_en"] = stoi(clean_norm_np, enhanced_norm_np, target_sr, extended=True)
        print(f"STOI en {res['stoi_en']}")
        res["stoi_no"] = stoi(clean_norm_np, noisy_norm_np, target_sr, extended=True)
        res["sdr_en"] = 10 * np.log10(np.sum(clean_norm_np**2) / (np.sum((clean_norm_np - enhanced_norm_np)**2) + 1e-8))
        res["sdr_no"] = 10 * np.log10(np.sum(clean_norm_np**2) / (np.sum((clean_norm_np - noisy_norm_np)**2) + 1e-8))
        res["si-sdr-en"] = compute_si_sdr(clean_norm_np, enhanced_norm_np)
        print(f"Si SDR en {res['si-sdr-en']}")
        res["si-sdr-no"] = compute_si_sdr(clean_norm_np, noisy_norm_np)
        
        try:
            res["pesq_en"] = pesq(target_sr, clean_norm_np, enhanced_norm_np, 'wb')
            print(res["pesq_en"] )
            res["pesq_no"] = pesq(target_sr, clean_norm_np, noisy_norm_np , 'wb')
        except:
            print("Warning: PESQ failed")
            res["pesq_en"] = None
            res["pesq_no"] = None
            
        enhanced_norm_np,_ = normalize_rms(enhanced_norm_np, target_rms=0.1)
        clean_norm_np,_ = normalize_rms(clean_norm_np, target_rms=0.1)
        noisy_norm_np,_ = normalize_rms(noisy_norm_np, target_rms=0.1)
          
        res["stoi_en_norm"] = stoi(clean_norm_np, enhanced_norm_np, target_sr, extended=True)
        res["stoi_no_norm"] = stoi(clean_norm_np, noisy_norm_np, target_sr, extended=True)
        res["sdr_en_norm"] = 10 * np.log10(np.sum(clean_norm_np**2) / (np.sum((clean_norm_np - enhanced_norm_np)**2) + 1e-8))
        res["sdr_no_norm"] = 10 * np.log10(np.sum(clean_norm_np**2) / (np.sum((clean_norm_np - noisy_norm_np)**2) + 1e-8))
        res["si_sdr_en_norm"] = compute_si_sdr(clean_norm_np, enhanced_norm_np)
        res["si_sdr_norm"] = compute_si_sdr(clean_norm_np, noisy_norm_np)
        
        try:
            res["pesq_en_norm"] = pesq(target_sr, clean_norm_np, enhanced_norm_np, 'wb')
            
            res["pesq_no_norm"] = pesq(target_sr, clean_norm_np, noisy_norm_np , 'wb')
        except:
            print("Warning: PESQ failed")
            res["pesq_en_norm"] = None
            res["pesq_no_norm"] = None  
        
    if config.compute_non_intrusive_metrics:
        scores_noisy = compute_squim(noisy_torch, squim_evaluator)
        scores_enhanced = compute_squim(enhanced_torch, squim_evaluator)
        res["squim-stoi-no"] = scores_noisy["stoi"]
        res["squim-pesq-no"] = scores_noisy["pesq"]
        res["squim-si-sdr-no"] = scores_noisy["si_sdr"]        
        res["squim-stoi-en"] = scores_enhanced["stoi"]
        res["squim-pesq-en"] = scores_enhanced["pesq"]
        res["squim-si-sdr-en"] = scores_enhanced["si_sdr"]    
    
    results.append(res)

def print_and_save_evaluation_results(output_path, results, tag=""):
    df = pd.DataFrame(results)
    
    filename = f"eval_results_{tag}.csv" 
    
    df.to_csv(output_path+"/"+filename,index = False)
    
    print("\n" + "=" * 50)
    print(f"{'Metric':<10} {'Enhanced':<15} {'Noisy':<15}")
    print("=" * 50)
    
    if config.compute_intrusive_metrics:    
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
    
    if config.compute_non_intrusive_metrics:
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
    
    alpha = np.dot(enhanced, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference

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
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        
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
    
    #print(f"Model loaded successfully from {MODEL_WEIGHTS}.")
    
    return model, checkpoint

def normalize_loudness(audio, meter, target_sr, target_lufs):
    gain_factor = get_loudness_norm_factor(audio, meter, target_sr, target_lufs)
    
    return gain_factor * audio

def normalize_rms(audio, target_rms, eps = 1e-8):
    current_rms = np.sqrt(np.mean(audio**2) + eps)
    
    scaling_factor = target_rms / (current_rms + eps)
    
    normalized_audio = audio * scaling_factor    
    
    return normalized_audio, scaling_factor




if __name__ == "__main__":
    
    # EXPECTED FOLDER STRUCTURE (code for testset B)
    # RECORDINGS/ 
    # ├── roof-rec-4th-testset/
    # │   ├── 20cm/
    # │   │   ├── dynamic/
    # │   │   │   └── segmented/
    # │   │   │       ├── clean/
    # │   │   │       ├── enhanced/
    # │   │   │       └── noisy/
    # │   │   └── shotgun/
    # │   │       └── segmented/
    # │   │           ├── clean/
    # │   │           ├── enhanced/
    # │   │           └── noisy/
    # │   └── 50cm/
    # │       ├── dynamic/
    # │       │   └── segmented/
    # │       │       ├── clean/
    # │       │       ├── enhanced/
    # │       │       └── noisy/
    # │       └── shotgun/
    # │           └── segmented/
    # │               ├── clean/
    # │               ├── enhanced/
    # │               └── noisy/
    BASE = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS"

    MODELS = {
        "no-overlap":  f"{BASE}/test_sets/TESTING/NO_OVERLAP/best_model_5.pth",
        "DNS":     f"{BASE}/test_sets/TESTING/DNS/best_model_DNS.pth",
        #"overlap": f"{BASE}/test_sets/TESTING/OVERLAP/best_model_overlap.pth",   
    }
    
    MODEL_LABELS = {
        "no-overlap": "Model A: Without Overlapping Segments",
        "DNS":        "DNS Model (Model B)",
        #"overlap":    "Model A: With Overlapping Segments",           
    }

    # Each entry: input_dir, clean_dir (None if non-intrusive only), gain_factor (for level correction)
    TESTSETS = {
      
        "testset_shotgun_50cm": {
            "input_dir":   f"{BASE}/roof-rec-4th-testset/50cm/shotgun/segmented/noisy",
            "clean_dir":   f"{BASE}/roof-rec-4th-testset/50cm/shotgun/segmented/clean",
            "output_dir":  f"{BASE}/roof-rec-4th-testset/50cm/shotgun/segmented/enhanced",
            "gain_factor": 0.3434,
        },
        "testset_shotgun_20cm": {                    
            "input_dir":   f"{BASE}/roof-rec-4th-testset/20cm/shotgun/segmented/noisy",
            "clean_dir":   f"{BASE}/roof-rec-4th-testset/20cm/shotgun/segmented/clean",
            "output_dir":  f"{BASE}/roof-rec-4th-testset/20cm/shotgun/segmented/enhanced",
            "gain_factor": 1.1152,
        },
        "testset_dynamic_50cm": {
            "input_dir":   f"{BASE}/roof-rec-4th-testset/50cm/dynamic/segmented/noisy",
            "clean_dir":   f"{BASE}/roof-rec-4th-testset/50cm/dynamic/segmented/clean",
            "output_dir":  f"{BASE}/roof-rec-4th-testset/50cm/dynamic/segmented/enhanced", 
            "gain_factor": 0.57277,
        },
        "testset_dynamic_20cm": {
            "input_dir":   f"{BASE}/roof-rec-4th-testset/20cm/dynamic/segmented/noisy",
            "clean_dir":   f"{BASE}/roof-rec-4th-testset/20cm/dynamic/segmented/clean",
            "output_dir":  f"{BASE}/roof-rec-4th-testset/20cm/dynamic/segmented/enhanced",
            "gain_factor": 0.70362,
         },
        
    }
    
    config = Config()
    
    filter_hp = butter(N=4, Wn=150, btype='highpass', fs=config.target_sr, output='sos')
    
    collected: dict = {}
    # loud_meter = pyln.Meter(config.target_sr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name, model_weights in MODELS.items():

        if not os.path.exists(model_weights):
            print(f"Model weights not found: {model_weights}")
            continue

        print(f"\n{'='*60}")
        print(f"  Loading model: {model_name}")
        print(f"{'='*60}")
        model, checkpoint = model_loader(model_weights, device)

        for ts_name, ts in TESTSETS.items():

            input_dir   = ts["input_dir"]
            clean_dir   = ts["clean_dir"]
            output_dir = ts["output_dir"]
            gain_factor = ts["gain_factor"]
            has_clean   = (clean_dir is not None)

            output_enhanced   = f"{output_dir}/{model_name}"
            noisy_normalized  = f"{input_dir}/noisy_normalized"
            os.makedirs(output_enhanced,  exist_ok=True)
            os.makedirs(noisy_normalized, exist_ok=True)

            if not os.path.exists(input_dir):
                print(f"Input dir not found: {input_dir}")
                continue

            files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
            
            if not files:
                print(f" No .wav files in {input_dir}")
                continue
            
            if config.compute_intrusive_metrics:
                cl_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]

            print(f"\n  Testset : {ts_name}  ({len(files)} files)")

            eval_results   = []
            squim_evaluator = SquimEvaluator()
            iteration = -1
            for file_name in files:
                iteration += 1
                input_path  = os.path.join(input_dir, file_name)
                base        = os.path.splitext(file_name)[0]
                output_name_enhanced = f"{base.replace('_noisy', '_enhanced')}.wav"
                
                output_enhanced_path  = os.path.join(output_enhanced,  output_name_enhanced)
                noisy_normalized_path = os.path.join(noisy_normalized, file_name)

                noisy_np, noisy_torch = load_audio(input_path, config.target_sr)

                print(f"    Processing: {file_name} -> {output_name_enhanced}")

                enhanced_torch = infer_complex_audio(
                    model=model,
                    noisy_np=noisy_np,
                    output_normalized_noisy=noisy_normalized_path,
                    output_normalized_enhanced=output_enhanced_path,
                )

                enhanced_np = enhanced_torch.squeeze(0).cpu().numpy()

                if config.compute_intrusive_metrics or config.compute_non_intrusive_metrics:
                    if config.compute_intrusive_metrics and has_clean:
                        clean_file = cl_files[iteration]
                        clean_wav_path = os.path.join(clean_dir, clean_file)
                        clean_np, _ = load_audio(clean_wav_path, config.target_sr)
                        print(f"    Reference: {clean_wav_path}")
                    else:
                        clean_np = 0

                    evaluate_file(
                        f_name=file_name,
                        enhanced_np=enhanced_np,
                        noisy_np=noisy_np,
                        noisy_torch=noisy_torch,
                        enhanced_torch=enhanced_torch,
                        squim_evaluator=squim_evaluator,
                        results=eval_results,
                        target_sr=config.target_sr,
                        clean_np=clean_np,
                        gain_factor=gain_factor,
                    )

                
               
                ####### OPTIONAL RMS LEVEL NORMALIZATION ##############
                # #Apply the filter forward and backward to ensure zero phase shift
                # noisy_np_filtered = sosfiltfilt(filter_hp, noisy_np)
                
                # _, scaling_factor = normalize_rms(noisy_np_filtered, target_rms = 0.04)
                
                # noisy_np = noisy_np * scaling_factor
                
                # if (config.compute_intrusive_metrics):
                #     clean_np,_ = normalize_rms(clean_np, target_rms = 0.04) 
                
                # if max(abs(enhanced_np) > 1.0) or max(abs(noisy_np) > 1.0):
             
                #     maximum = max(max(abs(enhanced_np)),max(abs(noisy_np)))
                #     enhanced_np = enhanced_np / maximum
                #     noisy_np = noisy_np / maximum
                    
                #     if config.compute_intrusive_metrics:
                #         clean_np = clean_np / maximum
                        
                #     #raise Warning("CLIPPING - scaling down")
                #     print(f"CLIPPING scaling down by 1/{maximum:.2f}")
                # sf.write(input_path, noisy_np,    config.target_sr)
                # sf.write(clean_wav_path, clean_np,    config.target_sr)
                
                
                sf.write(output_enhanced_path,  enhanced_np, config.target_sr)
                

            print(f"\n Done: {ts_name}, {model_name}")

            if config.compute_intrusive_metrics or config.compute_non_intrusive_metrics:
                print_and_save_evaluation_results(
                    input_dir,
                    eval_results,
                    tag=f"{ts_name}_{model_name}",  
                )
                
                avg = compute_average_metrics(eval_results)
                

                collected[(model_name, ts_name)] = {
                    "noisy": {
                        "STOI":   avg.get("stoi_no")          or avg.get("squim-stoi-no"),
                        "SI-SDR": avg.get("si-sdr-no")        or avg.get("squim-si-sdr-no"),
                        "PESQ":   avg.get("pesq_no")          or avg.get("squim-pesq-no"),
                    },
                    "enhanced": {
                        "STOI":   avg.get("stoi_en")          or avg.get("squim-stoi-en"),
                        "SI-SDR": avg.get("si-sdr-en")        or avg.get("squim-si-sdr-en"),
                        "PESQ":   avg.get("pesq_en")          or avg.get("squim-pesq-en"),
                    },
                    "enhanced_norm": {
                        "STOI-norm":   avg.get("stoi_en_norm"),
                        "SI-SDR-norm": avg.get("si-sdr-en_norm"),
                        "PESQ-norm":   avg.get("pesq_en_norm"),
                    },
                }
              
                save_to_csv(input_dir, collected)
              

            
            